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
#include <vector>
#include <omp.h>

namespace {

constexpr int input_size = 13, hidden_size = 16, steps = 4, states = 3;
constexpr int input_weights = input_size * hidden_size;
constexpr int recurrent_weights = hidden_size * hidden_size;
constexpr int gate_size = input_weights + recurrent_weights + hidden_size;
constexpr int outputs = 3 * gate_size, total_parameters = outputs + (hidden_size + 1) * states;
constexpr std::size_t batch_size = 512;

using Hidden = std::array<float, hidden_size>;
using Input = std::array<float, input_size>;
using Scores = std::array<float, states>;
using Parameters = std::array<float, total_parameters>;

struct Sequence {
    std::uint32_t length;
    std::array<std::uint32_t, states> truth;
    std::array<Input, steps> inputs;
};
static_assert(sizeof(Sequence) == 224);

struct Step {
    Input x;
    Hidden prev, update, reset, candidate, masked_prev, hidden;
};

struct Forward {
    std::array<Step, steps> cache;
    int length;
    Scores scores;
};

std::vector<Sequence> read_rows(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() <= 0 || in.tellg() % sizeof(Sequence)) throw std::runtime_error("bad data: " + path);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    std::vector<Sequence> rows(bytes / sizeof(Sequence));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(rows.data()), bytes)) throw std::runtime_error("short read: " + path);
    for (const auto& row : rows) {
        if (!row.length || row.length > steps || !std::accumulate(row.truth.begin(), row.truth.end(), 0u)) {
            throw std::runtime_error("invalid step or pulse count: " + path);
        }
        for (const auto& frame : row.inputs) if (!std::all_of(frame.begin(), frame.end(), [](float x) {
                return std::isfinite(x);
            })) throw std::runtime_error("nonfinite observation: " + path);
    }
    return rows;
}

Hidden affine(const float* p, const Input& x, const Hidden& prev) {
    Hidden out;
    std::copy_n(p + input_weights + recurrent_weights, hidden_size, out.begin());
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) out[j] += x[i] * p[i * hidden_size + j];
    }
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) out[j] += prev[i] * p[input_weights + i * hidden_size + j];
    }
    return out;
}

float sigmoid(float value) { return 1.f / (1.f + std::exp(-value)); }

Forward forward(const Parameters& p, const Sequence& row, bool with_history) {
    Forward result{};
    result.length = with_history ? row.length : 1;
    Hidden prev{};
    for (int i = 0; i < result.length; ++i) {
        auto& s = result.cache[i];
        s.x = row.inputs[with_history ? i : row.length - 1];
        if (!with_history) s.x[input_size - 2] = s.x[input_size - 1] = 0;
        s.prev = prev;
        s.update = affine(p.data(), s.x, prev);
        s.reset = affine(p.data() + gate_size, s.x, prev);
        for (int j = 0; j < hidden_size; ++j) {
            s.update[j] = sigmoid(s.update[j]);
            s.reset[j] = sigmoid(s.reset[j]);
            s.masked_prev[j] = s.reset[j] * prev[j];
        }
        s.candidate = affine(p.data() + 2 * gate_size, s.x, s.masked_prev);
        for (int j = 0; j < hidden_size; ++j) {
            s.candidate[j] = std::tanh(s.candidate[j]);
            s.hidden[j] = (1 - s.update[j]) * prev[j] + s.update[j] * s.candidate[j];
        }
        prev = s.hidden;
    }
    std::copy_n(p.data() + outputs + hidden_size * states, states, result.scores.begin());
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < states; ++j) result.scores[j] += prev[i] * p[outputs + i * states + j];
    }
    const float peak = *std::max_element(result.scores.begin(), result.scores.end());
    float total = 0;
    for (auto& value : result.scores) {
        value = std::exp(value - peak);
        total += value;
    }
    for (auto& value : result.scores) value /= total;
    return result;
}

Hidden backward_affine(const float* p, float* g, const Input& x, const Hidden& prev, const Hidden& delta) {
    Hidden h_gradient{};
    for (int j = 0; j < hidden_size; ++j) g[input_weights + recurrent_weights + j] += delta[j];
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) g[i * hidden_size + j] += x[i] * delta[j];
    }
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            const int index = input_weights + i * hidden_size + j;
            g[index] += prev[i] * delta[j];
            h_gradient[i] += p[index] * delta[j];
        }
    }
    return h_gradient;
}

void accumulate(const Parameters& p, Parameters& g, const Sequence& row, const Scores& weights, bool history) {
    const auto run = forward(p, row, history);
    const auto count = std::accumulate(row.truth.begin(), row.truth.end(), 0u);
    const float cap = static_cast<float>(std::min(count, 256u)) / count;
    Scores weighted{};
    for (int j = 0; j < states; ++j) weighted[j] = row.truth[j] * weights[j] * cap;
    const float total = std::accumulate(weighted.begin(), weighted.end(), 0.f);
    Hidden dh{};
    const auto& last = run.cache[run.length - 1].hidden;
    for (int j = 0; j < states; ++j) {
        const float delta = run.scores[j] * total - weighted[j];
        g[outputs + hidden_size * states + j] += delta;
        for (int i = 0; i < hidden_size; ++i) {
            g[outputs + i * states + j] += last[i] * delta;
            dh[i] += p[outputs + i * states + j] * delta;
        }
    }
    for (int i = run.length - 1; i >= 0; --i) {
        const auto& s = run.cache[i];
        Hidden dn, dz, dr;
        for (int j = 0; j < hidden_size; ++j) {
            dn[j] = dh[j] * s.update[j] * (1 - s.candidate[j] * s.candidate[j]);
            dz[j] = dh[j] * (s.candidate[j] - s.prev[j]) * s.update[j] * (1 - s.update[j]);
        }
        const auto dmasked = backward_affine(p.data() + 2 * gate_size, g.data() + 2 * gate_size, s.x, s.masked_prev, dn);
        for (int j = 0; j < hidden_size; ++j) dr[j] = dmasked[j] * s.prev[j] * s.reset[j] * (1 - s.reset[j]);
        const auto dr_prev = backward_affine(p.data() + gate_size, g.data() + gate_size, s.x, s.prev, dr);
        const auto dz_prev = backward_affine(p.data(), g.data(), s.x, s.prev, dz);
        for (int j = 0; j < hidden_size; ++j) {
            dh[j] = dh[j] * (1 - s.update[j]) + dmasked[j] * s.reset[j] + dr_prev[j] + dz_prev[j];
        }
    }
}

void save(const std::string& path, const Parameters& p, bool history) {
    std::ofstream out(path);
    out << "ZGY_GRU_STATE_1 " << history << '\n' << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (float value : p) out << value << '\n';
    if (!out) throw std::runtime_error("cannot write model: " + path);
}

double selection_f1(const Parameters& p, const std::vector<Sequence>& rows, bool history) {
    std::uint64_t counts[states * states]{};
    #pragma omp parallel for reduction(+:counts[:states * states]) schedule(static)
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto scores = forward(p, rows[i], history).scores;
        const auto pred = std::max_element(scores.begin(), scores.end()) - scores.begin();
        for (int j = 0; j < states; ++j) counts[j * states + pred] += rows[i].truth[j];
    }
    double f1 = 0;
    for (int i = 0; i < states; ++i) {
        double denominator = 0;
        for (int j = 0; j < states; ++j) denominator += counts[i * states + j] + counts[j * states + i];
        if (!denominator) throw std::runtime_error("absent selection state");
        f1 += 2. * counts[i * states + i] / denominator;
    }
    return f1 / states;
}

void train(const std::string& dev, const std::string& val, const std::string& model, bool history, int epochs, int seed) {
    const auto rows = read_rows(dev), selection = read_rows(val);
    Scores weights{}, counts{};
    for (const auto& row : rows) {
        const auto n = std::accumulate(row.truth.begin(), row.truth.end(), 0u);
        const float cap = static_cast<float>(std::min(n, 256u)) / n;
        for (int i = 0; i < states; ++i) counts[i] += row.truth[i] * cap;
    }
    const float largest = *std::max_element(counts.begin(), counts.end());
    for (int i = 0; i < states; ++i) {
        if (!counts[i]) throw std::runtime_error("absent fitting state");
        weights[i] = std::sqrt(largest / counts[i]);
    }
    std::mt19937 rng(seed);
    Parameters p{}, m{}, v{};
    std::uniform_real_distribution<float> uniform(-.20f, .20f);
    for (auto& x : p) x = uniform(rng);
    std::vector<std::size_t> order(rows.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<Parameters> gradients(omp_get_max_threads());
    float beta1 = 1, beta2 = 1;
    double best = -1.;
    int best_epoch = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        const float lr = .002f * (1.f - .65f * epoch / epochs);
        #pragma omp parallel
        {
            auto& local = gradients[omp_get_thread_num()];
            for (std::size_t start = 0; start < order.size(); start += batch_size) {
                const auto end = std::min(start + batch_size, order.size());
                local = {};
                #pragma omp for schedule(static)
                for (std::size_t i = start; i < end; ++i) accumulate(p, local, rows[order[i]], weights, history);
                #pragma omp single
                {
                    Parameters grad{};
                    for (const auto& row : gradients) {
                        for (int i = 0; i < total_parameters; ++i) grad[i] += row[i];
                    }
                    beta1 *= .9f;
                    beta2 *= .999f;
                    const float rate = lr * std::sqrt(1 - beta2) / (1 - beta1);
                    for (int i = 0; i < total_parameters; ++i) {
                        const float d = grad[i] / (end - start);
                        m[i] = .9f * m[i] + .1f * d;
                        v[i] = .999f * v[i] + .001f * d * d;
                        p[i] -= rate * m[i] / (std::sqrt(v[i]) + 1e-8f);
                    }
                }
            }
        }
        const double score = selection_f1(p, selection, history);
        std::cout << "epoch " << epoch + 1 << " macro_f1 " << score << std::endl;
        if (score > best) {
            best = score;
            best_epoch = epoch + 1;
            save(model, p, history);
        } else if (epoch + 1 - best_epoch >= 5) {
            break;
        }
    }
    if (!best_epoch) throw std::runtime_error("no model checkpoint");
    std::cout << "best_epoch " << best_epoch << " macro_f1 " << best << std::endl;
}

void predict(const std::string& model, const std::string& input, const std::string& output) {
    Parameters p;
    bool history;
    std::string header;
    std::ifstream in(model);
    if (!(in >> header >> history) || header != "ZGY_GRU_STATE_1") throw std::runtime_error("bad model: " + model);
    for (auto& value : p) in >> value;
    if (!in) throw std::runtime_error("truncated model: " + model);
    const auto rows = read_rows(input);
    std::vector<Scores> scores(rows.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < rows.size(); ++i) scores[i] = forward(p, rows[i], history).scores;
    std::ofstream file(output, std::ios::binary);
    if (!file.write(reinterpret_cast<const char*>(scores.data()), scores.size() * sizeof(Scores))) {
        throw std::runtime_error("cannot write predictions: " + output);
    }
}

void gradient_check() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(-.2f, .2f);
    Parameters p{}, g{};
    for (auto& value : p) value = uniform(rng);
    Sequence row{4, {0, 1, 0}, {}};
    for (auto& frame : row.inputs) for (auto& value : frame) value = 3 * uniform(rng);
    accumulate(p, g, row, {1, 1, 1}, true);
    constexpr float eps = .001f;
    float maximum = 0;
    for (int j = 0; j < total_parameters; ++j) {
        const float reference = p[j];
        p[j] = reference + eps;
        const float high = -std::log(forward(p, row, true).scores[1]);
        p[j] = reference - eps;
        const float low = -std::log(forward(p, row, true).scores[1]);
        p[j] = reference;
        maximum = std::max(maximum, std::abs(g[j] - (high - low) / (2 * eps)));
    }
    if (!(maximum < .001f)) throw std::runtime_error("GRU gradient check failed");
    if (forward(p, row, true).scores == forward(p, row, false).scores) throw std::runtime_error("history has no effect");
    std::cout << "maximum_gradient_error " << maximum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (command == "train" && argc == 8) {
            train(argv[2], argv[3], argv[4], std::stoi(argv[5]) != 0, std::stoi(argv[6]), std::stoi(argv[7]));
        } else if (command == "predict" && argc == 5) {
            predict(argv[2], argv[3], argv[4]);
        } else if (command == "selftest" && argc == 2) {
            gradient_check();
        } else {
            throw std::runtime_error("expected train, predict, or selftest");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
