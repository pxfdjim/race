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
#include <utility>
#include <vector>
#include <omp.h>

namespace {

constexpr int summary_dim = 43;
constexpr int pulse_dim = 7;
constexpr int pulse_hidden = 8;
constexpr int bands = 4;
constexpr int joined_dim = summary_dim + pulse_hidden * bands;
constexpr int hidden_dim = 32;
constexpr int classes = 3;
constexpr std::size_t batch_size = 512;

template<std::size_t N>
using Vector = std::array<float, N>;
using Pulse = Vector<pulse_dim>;
using Scores = Vector<classes>;

struct Sequence {
    Vector<summary_dim> summary;
    std::uint32_t offset, count;
    std::array<std::uint32_t, classes> truth;
};
static_assert(sizeof(Sequence) == 192 && sizeof(Pulse) == 28);

template<class T>
std::vector<T> read_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() <= 0 || in.tellg() % sizeof(T)) throw std::runtime_error("bad input: " + path);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    std::vector<T> values(bytes / sizeof(T));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(values.data()), bytes)) throw std::runtime_error("short read: " + path);
    return values;
}

void verify(const std::vector<Sequence>& rows, const std::vector<Pulse>& pulses, bool with_raw) {
    for (const auto& row : rows) {
        if (!row.count || std::accumulate(row.truth.begin(), row.truth.end(), 0u) != row.count ||
            (with_raw && static_cast<std::size_t>(row.offset) + row.count > pulses.size())) {
            throw std::runtime_error("invalid sequence count or pulse offset");
        }
    }
}

struct Weights {
    Vector<pulse_dim * pulse_hidden> pulse_w{};
    Vector<pulse_hidden> pulse_b{};
    Vector<joined_dim * hidden_dim> hidden_w{};
    Vector<hidden_dim> hidden_b{};
    Vector<hidden_dim * classes> output_w{};
    Scores output_b{};

    auto values() { return std::tie(pulse_w, pulse_b, hidden_w, hidden_b, output_w, output_b); }
    auto values() const { return std::tie(pulse_w, pulse_b, hidden_w, hidden_b, output_w, output_b); }
};

template<class F, class First, class... Rest, std::size_t... I>
void visit_arrays(F fn, std::index_sequence<I...>, First&& first, Rest&&... rest) {
    auto apply = [&](auto index) {
        constexpr auto i = decltype(index)::value;
        fn(std::get<i>(first), std::get<i>(rest)...);
    };
    (apply(std::integral_constant<std::size_t, I>{}), ...);
}

template<class F, class... W>
void visit(F fn, W&... weights) {
    visit_arrays(fn, std::make_index_sequence<6>{}, weights.values()...);
}

template<std::size_t In, std::size_t Out>
Vector<Out> dense(const Vector<In>& input, const Vector<In * Out>& matrix, const Vector<Out>& bias) {
    auto result = bias;
    for (std::size_t i = 0; i < In; ++i) {
        for (std::size_t j = 0; j < Out; ++j) result[j] += input[i] * matrix[i * Out + j];
    }
    return result;
}

template<std::size_t In, std::size_t Out>
Vector<In> backward(const Vector<In>& input, const Vector<Out>& delta, const Vector<In * Out>& matrix,
                    Vector<In * Out>& grad, Vector<Out>& bias) {
    Vector<In> previous{};
    for (std::size_t j = 0; j < Out; ++j) bias[j] += delta[j];
    for (std::size_t i = 0; i < In; ++i) {
        for (std::size_t j = 0; j < Out; ++j) {
            grad[i * Out + j] += input[i] * delta[j];
            previous[i] += matrix[i * Out + j] * delta[j];
        }
    }
    return previous;
}

int band(const Pulse& input) {
    return std::clamp(static_cast<int>(input[5] * bands), 0, bands - 1);
}

struct Forward {
    Vector<joined_dim> input{};
    Vector<hidden_dim> hidden;
    Scores probabilities;
    std::array<int, bands> band_counts{};
};

Forward forward(const Weights& model, const Sequence& row, const Pulse* raw) {
    Forward out;
    std::copy(row.summary.begin(), row.summary.end(), out.input.begin());
    if (raw) {
        for (std::size_t i = row.offset, end = i + row.count; i < end; ++i) {
            const auto& p = raw[i];
            const int group = band(p);
            ++out.band_counts[group];
            const auto h = dense(p, model.pulse_w, model.pulse_b);
            for (int j = 0; j < pulse_hidden; ++j) {
                out.input[summary_dim + group * pulse_hidden + j] += std::max(0.f, h[j]);
            }
        }
        for (int group = 0; group < bands; ++group) {
            const float inverse = 1.f / std::max(1, out.band_counts[group]);
            for (int j = 0; j < pulse_hidden; ++j) out.input[summary_dim + group * pulse_hidden + j] *= inverse;
        }
    }
    out.hidden = dense(out.input, model.hidden_w, model.hidden_b);
    for (auto& h : out.hidden) h = std::max(0.f, h);
    out.probabilities = dense(out.hidden, model.output_w, model.output_b);
    const float peak = *std::max_element(out.probabilities.begin(), out.probabilities.end());
    float total = 0;
    for (auto& x : out.probabilities) {
        x = std::exp(x - peak);
        total += x;
    }
    for (auto& x : out.probabilities) x /= total;
    return out;
}

void accumulate(const Weights& model, const Sequence& row, const Pulse* raw, const Scores& class_weights,
                Weights& gradient) {
    const auto activations = forward(model, row, raw);
    const float count_factor = static_cast<float>(std::min(row.count, 256u)) / row.count;
    Scores weighted_counts{};
    for (int i = 0; i < classes; ++i) weighted_counts[i] = class_weights[i] * row.truth[i] * count_factor;
    const float weight = std::accumulate(weighted_counts.begin(), weighted_counts.end(), 0.f);
    auto delta = activations.probabilities;
    for (int i = 0; i < classes; ++i) delta[i] = delta[i] * weight - weighted_counts[i];
    auto hidden_delta = backward(activations.hidden, delta, model.output_w, gradient.output_w, gradient.output_b);
    for (int i = 0; i < hidden_dim; ++i) if (activations.hidden[i] <= 0) hidden_delta[i] = 0;
    const auto input_delta = backward(activations.input, hidden_delta, model.hidden_w, gradient.hidden_w, gradient.hidden_b);
    if (!raw) return;
    for (std::size_t i = row.offset, end = i + row.count; i < end; ++i) {
        const auto& p = raw[i];
        const int group = band(p);
        const auto h = dense(p, model.pulse_w, model.pulse_b);
        const float inverse = 1.f / activations.band_counts[group];
        for (int j = 0; j < pulse_hidden; ++j) {
            if (h[j] <= 0) continue;
            const float d = input_delta[summary_dim + group * pulse_hidden + j] * inverse;
            gradient.pulse_b[j] += d;
            for (int k = 0; k < pulse_dim; ++k) gradient.pulse_w[k * pulse_hidden + j] += d * p[k];
        }
    }
}

void save(const std::string& path, bool use_raw, const Weights& model) {
    std::ofstream out(path);
    out << "ZGY_PULSE_STATE_1 " << use_raw << '\n' << std::setprecision(std::numeric_limits<float>::max_digits10);
    visit([&](const auto& values) {
        for (float value : values) out << value << ' ';
        out << '\n';
    }, model);
    if (!out) throw std::runtime_error("cannot save model: " + path);
}

std::pair<Weights, bool> load(const std::string& path) {
    std::ifstream in(path);
    std::string format;
    bool use_raw;
    Weights model;
    if (!(in >> format >> use_raw) || format != "ZGY_PULSE_STATE_1") throw std::runtime_error("bad model: " + path);
    visit([&](auto& values) { for (float& value : values) in >> value; }, model);
    if (!in) throw std::runtime_error("truncated model: " + path);
    return {model, use_raw};
}

double selection_f1(const Weights& model, const std::vector<Sequence>& selection, const Pulse* raw) {
    std::uint64_t cells[classes * classes]{};
    #pragma omp parallel for reduction(+:cells[:classes * classes]) schedule(static)
    for (std::size_t i = 0; i < selection.size(); ++i) {
        const auto& row = selection[i];
        const auto p = forward(model, row, raw).probabilities;
        const auto label = std::max_element(p.begin(), p.end()) - p.begin();
        for (int y = 0; y < classes; ++y) cells[y * classes + label] += row.truth[y];
    }
    double f1 = 0.;
    for (int i = 0; i < classes; ++i) {
        double denominator = 0.;
        for (int j = 0; j < classes; ++j) denominator += cells[i * classes + j] + cells[j * classes + i];
        if (!denominator) throw std::runtime_error("selection has an absent state");
        f1 += (2. * cells[i * classes + i]) / denominator;
    }
    return f1 / classes;
}

void train(const std::string& pulse_file, const std::string& training_file, const std::string& selection_file,
           const std::string& output, bool use_raw, int epochs, int seed) {
    if (epochs < 1) throw std::runtime_error("expected a positive epoch count");
    const auto training = read_binary<Sequence>(training_file), selection = read_binary<Sequence>(selection_file);
    const auto pulses = use_raw ? read_binary<Pulse>(pulse_file) : std::vector<Pulse>{};
    verify(training, pulses, use_raw);
    verify(selection, pulses, use_raw);
    const Pulse* raw = use_raw ? pulses.data() : nullptr;
    Scores class_weights{}, class_counts{};
    for (const auto& row : training) {
        const float factor = static_cast<float>(std::min(row.count, 256u)) / row.count;
        for (int i = 0; i < classes; ++i) class_counts[i] += row.truth[i] * factor;
    }
    const float largest = *std::max_element(class_counts.begin(), class_counts.end());
    for (int i = 0; i < classes; ++i) {
        if (class_counts[i] <= 0) throw std::runtime_error("development has an absent state");
        class_weights[i] = std::sqrt(largest / class_counts[i]);
    }
    std::mt19937 rng(seed);
    Weights model, first_moment, second_moment;
    auto init = [&](auto& values, int input, int output_size) {
        const float range = std::sqrt(6.f / (input + output_size));
        std::uniform_real_distribution<float> uniform(-range, range);
        for (auto& value : values) value = uniform(rng);
    };
    init(model.pulse_w, pulse_dim, pulse_hidden);
    init(model.hidden_w, joined_dim, hidden_dim);
    init(model.output_w, hidden_dim, classes);
    std::vector<std::size_t> order(training.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<Weights> gradients(omp_get_max_threads());
    float decay1 = 1.f, decay2 = 1.f;
    double best = -1.;
    int best_epoch = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        const float rate = .0025f * (1.f - .65f * epoch / epochs);
        #pragma omp parallel
        {
            auto& local = gradients[omp_get_thread_num()];
            for (std::size_t start = 0; start < order.size(); start += batch_size) {
                const std::size_t end = std::min(start + batch_size, order.size());
                local = {};
                #pragma omp for schedule(static)
                for (std::size_t i = start; i < end; ++i) accumulate(model, training[order[i]], raw, class_weights, local);
                #pragma omp single
                {
                    Weights combined{};
                    for (const auto& g : gradients) {
                        visit([](auto& a, const auto& b) {
                            for (std::size_t j = 0; j < a.size(); ++j) a[j] += b[j];
                        }, combined, g);
                    }
                    decay1 *= .9f;
                    decay2 *= .999f;
                    const float corrected = rate * std::sqrt(1 - decay2) / (1 - decay1);
                    const float step_scale = 1.f / static_cast<float>(end - start);
                    visit([&](auto& values, auto& m, auto& v, const auto& grad) {
                        for (std::size_t j = 0; j < values.size(); ++j) {
                            const float d = grad[j] * step_scale;
                            m[j] = .9f * m[j] + .1f * d;
                            v[j] = .999f * v[j] + .001f * d * d;
                            values[j] -= corrected * m[j] / (std::sqrt(v[j]) + 1e-8f);
                        }
                    }, model, first_moment, second_moment, combined);
                }
            }
        }
        const double score = selection_f1(model, selection, raw);
        std::cout << "epoch " << epoch + 1 << " selection_macro_f1 " << score << std::endl;
        if (score > best) {
            best = score;
            best_epoch = epoch + 1;
            save(output, use_raw, model);
        } else if (epoch + 1 - best_epoch >= 5) {
            break;
        }
    }
    std::cout << "best_epoch " << best_epoch << " selection_macro_f1 " << best << std::endl;
}

void predict(const std::string& pulse_file, const std::string& input, const std::string& path,
             const std::string& output) {
    const auto [model, use_raw] = load(path);
    const auto rows = read_binary<Sequence>(input);
    const auto pulses = use_raw ? read_binary<Pulse>(pulse_file) : std::vector<Pulse>{};
    verify(rows, pulses, use_raw);
    const Pulse* raw = use_raw ? pulses.data() : nullptr;
    std::vector<Scores> scores(rows.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < rows.size(); ++i) scores[i] = forward(model, rows[i], raw).probabilities;
    std::ofstream stream(output, std::ios::binary);
    if (!stream.write(reinterpret_cast<const char*>(scores.data()), scores.size() * sizeof(Scores))) {
        throw std::runtime_error("cannot save scores: " + output);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (command == "train" && argc == 9) {
            train(argv[2], argv[3], argv[4], argv[5], std::stoi(argv[6]) != 0, std::stoi(argv[7]), std::stoi(argv[8]));
        } else if (command == "predict" && argc == 6) {
            predict(argv[2], argv[3], argv[4], argv[5]);
        } else {
            throw std::runtime_error("expected train or predict with its file arguments");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
