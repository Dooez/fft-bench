#include "common.hpp"
#include "glaze/glaze.hpp"
#include "kfr_runner.hpp"
#include "pcx_runner.hpp"

#include <algorithm>
#include <chrono>
#include <fftw3.h>
#include <print>
#include <pthread.h>
#include <random>

namespace chrono = std::chrono;
namespace stdr   = std::ranges;
namespace stdv   = std::views;

using nano_s  = chrono::duration<f64, std::nano>;
using micro_s = chrono::duration<f64, std::micro>;
using seconds = chrono::duration<f64>;

template<typename T>
concept any_runner = true;
namespace {
auto measure_inplace_iteration(const auto& runner, auto preheat_view, auto data_view) {
    auto n = data_view.size();
    for (auto&& d: preheat_view) {
        runner.fft(d);
    }
    auto start = chrono::high_resolution_clock::now();
    for (auto&& d: data_view) {
        runner.fft(d);
    }
    auto end      = chrono::high_resolution_clock::now();
    auto duration = nano_s(end - start);
    return duration / n;
}
struct bench_params {
    uZ preheat_cnt         = 16;
    uZ min_data_size_bytes = 4UZ * 1024UZ * 1024UZ * 4;

    f64 filter_sd_coeff = 5;    // discard durations that do not lie within mean ± 5 * sd

    f64 t_factor             = 2.9;     // approximate t_factor for big n and 99.5% confidence
    f64 confidence_delta     = 0.01;    // target UCL of 100.1% of mean
    f64 min_confidence_delta = 0.05;    // minimum UCL of 102% of mean

    micro_s min_single_measurement{300U};
    seconds min_experiment{0.};
    seconds soft_max_experiment{std::numeric_limits<f64>::max()};
};

template<any_runner Runner>
auto measure_size(uZ size, bench_params param = {}) {
    using real             = Runner::real;
    using data_t           = std::vector<std::complex<real>>;
    const auto preheat_cnt = param.preheat_cnt;
    const auto runner      = Runner(size);

    auto init_data = [&](data_t& data) { data.resize(size); };

    auto big_data_cnt = ((param.min_data_size_bytes / sizeof(real) / 2) + size - 1) / size;
    big_data_cnt      = std::max(big_data_cnt, 1UZ);
    auto big_data     = std::vector<data_t>(big_data_cnt);
    for (auto&& data: big_data)
        init_data(data);
    auto data_getter = stdv::transform([&](auto i) -> auto& { return big_data[i]; });

    auto random_idxs    = std::vector<uZ>(std::max(big_data_cnt * 4, preheat_cnt + 1));
    auto randomize_idxs = [&, rd = std::mt19937()] mutable {
        for (auto [i, v]: stdv::enumerate(random_idxs))
            v = i % big_data_cnt;
        auto dist = std::uniform_int_distribution<uZ>(0, random_idxs.size() - 1);
        for (auto v: random_idxs)
            std::swap(v, random_idxs[dist(rd)]);
    };
    randomize_idxs();

    uZ meas_cnt = 1;
    while (true) {
        auto preheat_span = random_idxs | stdv::take(preheat_cnt) | data_getter;
        auto data_view    = random_idxs | stdv::drop(preheat_cnt) | stdv::take(meas_cnt) | data_getter;
        auto avg_dur      = measure_inplace_iteration(runner, preheat_span, data_view);
        auto meas_dur     = avg_dur * meas_cnt;
        if (meas_dur < param.min_single_measurement) {
            meas_cnt *= 2;
            if (random_idxs.size() < meas_cnt + preheat_cnt) {
                random_idxs.resize(meas_cnt + preheat_cnt);
                randomize_idxs();
            }
        } else
            break;
    }

    auto durations = std::vector<nano_s>();

    struct stats_t {
        nano_s mean_dur;
        f64    sd;            // standard deviation
        f64    conf_delta;    // half-size of confidence interval
    } stats{};
    auto update_stats = [&] {
        using std::plus;
        auto o_mean_dur = stdr::fold_left_first(durations, plus<nano_s>());
        if (!o_mean_dur)
            throw std::runtime_error("Error measuring mean");

        auto mean_dur = *o_mean_dur / durations.size();
        auto variance = stdr::fold_left(durations,
                                        0.,
                                        [=](auto a, auto b) {
                                            auto c = (b - mean_dur).count();
                                            return a + c * c;
                                        })
                        / durations.size();
        auto sd = std::sqrt(variance);

        auto delta = param.t_factor * sd / std::sqrt(static_cast<f64>(durations.size()));
        stats      = {mean_dur, sd, delta};
    };

    auto filter_data = [&] {
        auto filtered_durations = std::vector<nano_s>();
        filtered_durations.reserve(durations.size());
        auto out = std::back_inserter(filtered_durations);
        stdr::copy_if(durations, out, [&](auto dur) {
            auto v    = dur.count();
            auto mean = stats.mean_dur.count();
            return !(std::isnan(v)                                     //
                     || v > mean + stats.sd * param.filter_sd_coeff    //
                     || v < mean - stats.sd * param.filter_sd_coeff);
        });
        using std::swap;
        swap(filtered_durations, durations);
    };

    auto experiment_finished = [&] {
        auto total_dur = stats.mean_dur * durations.size() * meas_cnt;
        if (total_dur < param.min_experiment)
            return false;
        if (total_dur >= param.soft_max_experiment)
            return true;
        if (stats.conf_delta / stats.mean_dur.count() < param.confidence_delta)
            return true;
        return false;
    };

    durations.reserve(random_idxs.size() / meas_cnt * 4);
    uZ repeats = 1;
    while (true) {
        for (auto i: stdv::iota(repeats / 2, repeats)) {
            for (auto sample_idxs: random_idxs | stdv::chunk(meas_cnt + preheat_cnt)) {
                auto preheat_view = sample_idxs | stdv::take(preheat_cnt) | data_getter;
                auto data_view    = sample_idxs | stdv::drop(preheat_cnt) | data_getter;
                auto dur          = measure_inplace_iteration(runner, preheat_view, data_view);
                if (!std::isinf(dur.count()))
                    durations.push_back(dur);
            }
        }
        update_stats();
        if (stats.conf_delta / stats.mean_dur.count() < param.min_confidence_delta) {
            filter_data();
            update_stats();
            if (experiment_finished())
                break;
        } else {
            repeats *= 2;
        }
    }

    std::print("{} measurements of {} samples\n", durations.size(), meas_cnt);
    auto delta_perc = stats.conf_delta / stats.mean_dur.count() * 100;
    std::print("Mean: {}, standard deviation: {:.3}, t: {:.2}%\n", stats.mean_dur, stats.sd, delta_perc);
    auto [min, max] = stdr::minmax_element(durations);
    std::print("Min: {}, max: {}\n", *min, *max);
    auto perf_metric = 1 / (stats.mean_dur.count() / size / std::log2(size));
    std::print("Metric: {}\n", perf_metric);
    return stats.mean_dur.count();
}    // namespace


}    // namespace
auto main() -> int {
    pthread_t current_thread = pthread_self();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);    // Set affinity to the specified core_id
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::println("Error calling pthread_setaffinity_np: {}", rc);
        return -1;
    }
    constexpr auto prm = bench_params{
        .min_data_size_bytes = 4UZ * 2048 * 2 * 32,
        .confidence_delta    = 0.001,
        .min_experiment      = seconds{10},
        .soft_max_experiment = seconds{30},
    };
    auto pcx_results  = std::vector<f64>{};
    auto kfr_results  = std::vector<f64>{};
    uZ   fft_size     = 128;
    uZ   fft_end_size = 8192UZ * 8;
    while (fft_size <= fft_end_size) {
        std::println("pcx:");
        pcx_results.push_back(measure_size<pcx_runner<f32>>(fft_size, prm));
        std::println("kfr:");
        kfr_results.push_back(measure_size<kfr_runner<f32>>(fft_size, prm));
        fft_size *= 2;
    }
    (void)glz::write_file_csv(pcx_results, "results/l1cache_pcx_results.csv", std::string{});
    (void)glz::write_file_csv(kfr_results, "results/l1cache_kfr_results.csv", std::string{});

    return 0;
}
