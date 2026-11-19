#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <omp.h>

namespace {

constexpr int classes = 8;
constexpr int rf_bins = 12;
constexpr int inputs = 7 + rf_bins;
constexpr int hidden = 40;
constexpr int hidden2 = 24;
constexpr int batch_size = 1024;

struct Row {
    float rf, pw, pri, spread, mode;
    float label, weight, sequence_index, statistical;
};
static_assert(sizeof(Row) == 9 * sizeof(float));
using Input = std::array<float, inputs>;
using Scores = std::array<float, classes>;

std::vector<Row> read_rows(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (!bytes || bytes % sizeof(Row)) throw std::runtime_error("invalid data size: " + path);
    std::vector<Row> rows(bytes / sizeof(Row));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(rows.data()), bytes)) throw std::runtime_error("short read: " + path);
    return rows;
}

struct Parameters {
    std::array<float, inputs * hidden> w1{};
    std::array<float, hidden> b1{};
    std::array<float, hidden * hidden2> w2{};
    std::array<float, hidden2> b2{};
    std::array<float, hidden2 * classes> w3{};
    Scores b3{};

    auto arrays() { return std::tie(w1, b1, w2, b2, w3, b3); }
    auto arrays() const { return std::tie(w1, b1, w2, b2, w3, b3); }
};

template<class F, class Tuple, std::size_t... I, class... Rest>
void for_arrays(F&& operation, std::index_sequence<I...>, Tuple&& first, Rest&&... rest) {
    auto apply_one = [&](auto index) {
        constexpr auto i = decltype(index)::value;
        operation(std::get<i>(first), std::get<i>(rest)...);
    };
    (apply_one(std::integral_constant<std::size_t, I>{}), ...);
}

template<class F, class... P>
void visit_parameters(F&& operation, P&... parameters) {
    for_arrays(operation, std::make_index_sequence<6>{}, parameters.arrays()...);
}

struct Encoding {
    std::array<float, rf_bins> rf_centers{};

    void fit(const std::vector<Row>& rows) {
        std::vector<float> rf;
        rf.reserve(rows.size());
        for (const auto& row : rows) rf.push_back(row.rf);
        std::sort(rf.begin(), rf.end());
        for (int i = 0; i < rf_bins; ++i) rf_centers[i] = rf[(2 * i + 1) * rf.size() / (2 * rf_bins)];
    }

    Input operator()(const Row& row, bool hide_pri = false) const {
        Input x{};
        x[0] = (row.rf - 32000.f) / 4000.f;
        x[1] = row.pw / 2.f;
        for (int i = 0; i < rf_bins; ++i) {
            x[7 + i] = std::max(0.f, 1.f - std::abs(row.rf - rf_centers[i]) / 100.f);
        }
        if (row.mode == 0 || hide_pri) {
            x[6] = 1.f;
        } else {
            x[2] = std::log(row.pri / 300.f);
            x[3] = std::log1p(20.f * row.spread);
            x[4] = row.mode == 1;
            x[5] = row.mode == 2;
        }
        return x;
    }
};

struct LayerState {
    std::array<float, hidden> a1;
    std::array<float, hidden2> a2;
    Scores probabilities;
};

template<std::size_t In, std::size_t Out>
std::array<float, Out> dense(const std::array<float, In>& x,
                             const std::array<float, In * Out>& w,
                             const std::array<float, Out>& bias) {
    auto result = bias;
    for (std::size_t i = 0; i < In; ++i) {
        for (std::size_t j = 0; j < Out; ++j) result[j] += x[i] * w[i * Out + j];
    }
    return result;
}

LayerState forward(const Parameters& p, const Input& x) {
    LayerState result;
    result.a1 = dense(x, p.w1, p.b1);
    for (auto& value : result.a1) value = std::max(0.f, value);
    result.a2 = dense(result.a1, p.w2, p.b2);
    for (auto& value : result.a2) value = std::max(0.f, value);
    result.probabilities = dense(result.a2, p.w3, p.b3);
    const float peak = *std::max_element(result.probabilities.begin(), result.probabilities.end());
    float sum = 0;
    for (auto& value : result.probabilities) {
        value = std::exp(value - peak);
        sum += value;
    }
    for (auto& value : result.probabilities) value /= sum;
    return result;
}

template<std::size_t In, std::size_t Out>
std::array<float, In> backward(const std::array<float, In>& x,
                               const std::array<float, Out>& delta,
                               const std::array<float, In * Out>& weights,
                               std::array<float, In * Out>& grad,
                               std::array<float, Out>& grad_bias) {
    std::array<float, In> previous{};
    for (std::size_t j = 0; j < Out; ++j) grad_bias[j] += delta[j];
    for (std::size_t i = 0; i < In; ++i) {
        for (std::size_t j = 0; j < Out; ++j) {
            const auto index = i * Out + j;
            grad[index] += x[i] * delta[j];
            previous[i] += weights[index] * delta[j];
        }
    }
    return previous;
}

void accumulate(const Parameters& p, const Input& x, int label, float weight, Parameters& g) {
    const auto activations = forward(p, x);
    auto delta3 = activations.probabilities;
    delta3[label] -= 1.f;
    for (auto& v : delta3) v *= weight;
    auto delta2 = backward(activations.a2, delta3, p.w3, g.w3, g.b3);
    for (int i = 0; i < hidden2; ++i) if (activations.a2[i] <= 0) delta2[i] = 0;
    auto delta1 = backward(activations.a1, delta2, p.w2, g.w2, g.b2);
    for (int i = 0; i < hidden; ++i) if (activations.a1[i] <= 0) delta1[i] = 0;
    backward(x, delta1, p.w1, g.w1, g.b1);
}

int best_class(const Scores& scores) {
    return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

void save(const std::string& path, const Encoding& encoding, const Parameters& p) {
    std::ofstream out(path);
    out << "ZGY_MLP_1 " << inputs << ' ' << hidden << ' ' << hidden2 << ' ' << classes << '\n';
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (auto center : encoding.rf_centers) out << center << ' ';
    out << '\n';
    visit_parameters([&](const auto& values) {
        for (auto x : values) out << x << ' ';
        out << '\n';
    }, p);
    if (!out) throw std::runtime_error("cannot save model: " + path);
}

std::pair<Encoding, Parameters> load(const std::string& path) {
    std::ifstream in(path);
    std::string header;
    int a, b, c, d;
    if (!(in >> header >> a >> b >> c >> d) || header != "ZGY_MLP_1" ||
            a != inputs || b != hidden || c != hidden2 || d != classes) {
        throw std::runtime_error("invalid model: " + path);
    }
    Encoding encoding;
    Parameters parameters;
    for (auto& x : encoding.rf_centers) in >> x;
    visit_parameters([&](auto& values) { for (auto& x : values) in >> x; }, parameters);
    if (!in) throw std::runtime_error("truncated model: " + path);
    return {encoding, parameters};
}

float macro_f1(const std::vector<Row>& validation, const std::vector<Input>& features,
               const Parameters& p) {
    double cells[classes * classes]{};
    #pragma omp parallel for reduction(+:cells[:classes * classes]) schedule(static)
    for (std::size_t i = 0; i < validation.size(); ++i) {
        const auto& r = validation[i];
        const int pred = best_class(forward(p, features[i]).probabilities);
        cells[static_cast<int>(r.label) * classes + pred] += r.weight;
    }
    double f1 = 0.;
    int present = 0;
    for (int a = 0; a < classes; ++a) {
        double predicted = 0., actual = 0.;
        for (int b = 0; b < classes; ++b) {
            actual += cells[a * classes + b];
            predicted += cells[b * classes + a];
        }
        if (actual > 0.) {
            f1 += 2 * cells[a * classes + a] / (actual + predicted);
            ++present;
        }
    }
    return f1 / present;
}

void train(const std::string& training_path, const std::string& validation_path,
           const std::string& output, int epochs, int seed, float pri_dropout) {
    const auto rows = read_rows(training_path);
    const auto validation = read_rows(validation_path);
    Encoding encoding;
    encoding.fit(rows);
    std::vector<Input> x, masked, val_x;
    for (const auto& r : rows) {
        x.push_back(encoding(r));
        if (pri_dropout > 0) masked.push_back(encoding(r, true));
    }
    for (const auto& r : validation) val_x.push_back(encoding(r));
    std::mt19937 rng(seed);
    Parameters p, first, second;
    auto init = [&](auto& w, int fan_in, int fan_out) {
        const float limit = std::sqrt(6.f / (fan_in + fan_out));
        std::uniform_real_distribution<float> uniform(-limit, limit);
        for (auto& value : w) value = uniform(rng);
    };
    init(p.w1, inputs, hidden);
    init(p.w2, hidden, hidden2);
    init(p.w3, hidden2, classes);
    std::vector<std::size_t> order(rows.size());
    std::iota(order.begin(), order.end(), 0);
    const int threads = omp_get_max_threads();
    std::vector<Parameters> gradients(threads);
    std::vector<float> weights(threads);
    float best = -1.f, decay1 = 1.f, decay2 = 1.f;
    int best_epoch = 0;
    const auto dropout_limit = static_cast<std::uint32_t>(pri_dropout * 1000);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        const float lr = .0025f * (1.f - .65f * epoch / epochs);
        #pragma omp parallel
        {
            const int thread = omp_get_thread_num();
            for (std::size_t start = 0; start < rows.size(); start += batch_size) {
                const std::size_t end = std::min(start + batch_size, rows.size());
                auto& g = gradients[thread];
                g = {};
                weights[thread] = 0;
                #pragma omp for schedule(static)
                for (std::size_t pos = start; pos < end; ++pos) {
                    const auto i = order[pos];
                    const auto& r = rows[i];
                    const auto mask = (static_cast<std::uint32_t>(i) * 2654435761u
                                       ^ static_cast<std::uint32_t>(epoch + seed) * 2246822519u) % 1000;
                    accumulate(p, mask < dropout_limit ? masked[i] : x[i],
                               static_cast<int>(r.label), r.weight, g);
                    weights[thread] += r.weight;
                }
                #pragma omp single
                {
                    const float total = std::accumulate(weights.begin(), weights.end(), 0.f);
                    Parameters gsum = gradients[0];
                    for (int t = 1; t < threads; ++t) {
                        visit_parameters([](auto& dest, const auto& from) {
                            for (std::size_t i = 0; i < dest.size(); ++i) dest[i] += from[i];
                        }, gsum, gradients[t]);
                    }
                    decay1 *= .9f;
                    decay2 *= .999f;
                    const float rate = lr * std::sqrt(1.f - decay2) / (1.f - decay1);
                    visit_parameters([&](auto& v, auto& m, auto& s, const auto& grad) {
                        for (std::size_t i = 0; i < v.size(); ++i) {
                            const float d = grad[i] / total;
                            m[i] = .9f * m[i] + .1f * d;
                            s[i] = .999f * s[i] + .001f * d * d;
                            v[i] -= rate * m[i] / (std::sqrt(s[i]) + 1e-8f);
                        }
                    }, p, first, second, gsum);
                }
            }
        }
        const float score = macro_f1(validation, val_x, p);
        std::cerr << output << " epoch " << epoch + 1 << " selection_macro_f1 " << score << '\n';
        if (score > best) {
            save(output, encoding, p);
            best = score;
            best_epoch = epoch + 1;
        } else if (epoch + 1 - best_epoch >= 6) {
            break;
        }
    }
    std::cout << "selection best epoch " << best_epoch << " macro_f1 " << best << '\n';
}

void evaluate(const std::string& pulse_model, const std::string& sequence_model,
              const std::string& pulse_file, const std::string& sequence_file) {
    const auto [pulse_encoder, pulse] = load(pulse_model);
    const auto [sequence_encoder, sequence] = load(sequence_model);
    const auto sequences = read_rows(sequence_file);
    const auto rows = read_rows(pulse_file);
    std::vector<int> seq_predictions(sequences.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < sequences.size(); ++i) {
        seq_predictions[i] = best_class(forward(sequence, sequence_encoder(sequences[i])).probabilities);
    }
    std::uint64_t cells[3 * 2 * classes * classes]{};
    #pragma omp parallel for reduction(+:cells[:3 * 2 * classes * classes]) schedule(static)
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        const int seq = static_cast<int>(r.sequence_index);
        const int group = seq >= 0;
        const int pulse_label = best_class(forward(pulse, pulse_encoder(r)).probabilities);
        const int preds[] = {static_cast<int>(r.statistical), pulse_label,
                             group ? seq_predictions[seq] : pulse_label};
        for (int m = 0; m < 3; ++m) {
            ++cells[((m * 2 + group) * classes + static_cast<int>(r.label)) * classes + preds[m]];
        }
    }
    for (int m = 0; m < 3; ++m) for (int g = 0; g < 2; ++g) {
        for (int a = 0; a < classes; ++a) for (int b = 0; b < classes; ++b) {
            std::cout << m << ' ' << g << ' ' << a << ' ' << b << ' '
                      << cells[((m * 2 + g) * classes + a) * classes + b] << '\n';
        }
    }
}

void predict(const std::string& path, const std::string& input, const std::string& output) {
    const auto [encoding, model] = load(path);
    const auto rows = read_rows(input);
    std::vector<Scores> scores(rows.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < rows.size(); ++i) scores[i] = forward(model, encoding(rows[i])).probabilities;
    std::ofstream out(output, std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(scores.data()), scores.size() * sizeof(Scores))) {
        throw std::runtime_error("cannot save scores: " + output);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (command == "train" && argc == 8) train(argv[2], argv[3], argv[4], std::stoi(argv[5]),
                                                   std::stoi(argv[6]), std::stof(argv[7]));
        else if (command == "evaluate" && argc == 6) evaluate(argv[2], argv[3], argv[4], argv[5]);
        else if (command == "predict" && argc == 5) predict(argv[2], argv[3], argv[4]);
        else throw std::runtime_error("expected train, evaluate, or predict with their file arguments");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
