#include "common.hpp"
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
struct measurement_params {
    uZ minimum_iterations  = 1000;
    uZ preheat_cnt         = 1;
    uZ min_data_size_bytes = 4UZ * 1024UZ * 1024UZ * 4;

    micro_s min_measurement{300U};
};

template<any_runner Runner>
auto measure_size(uZ size, measurement_params param = {}) {
    using real       = Runner::real;
    using cxvector   = std::vector<std::complex<real>>;
    auto runner      = Runner(size);
    auto results     = std::vector<f64>{};
    auto preheat_cnt = param.preheat_cnt;

    auto big_data_cnt    = ((param.min_data_size_bytes / sizeof(real) / 2) + size - 1) / size;
    big_data_cnt         = std::max(big_data_cnt, preheat_cnt + 1UZ);
    auto big_data        = std::vector<cxvector>();
    auto resize_big_data = [&](uZ new_size) {
        auto old_size = big_data.size();
        big_data.resize(new_size);
        for (auto&& data: big_data | stdv::drop(old_size)) {
            data.resize(size);
            // potentially fill
        }
    };
    resize_big_data(big_data_cnt);
    auto big_data_getter = stdv::transform([&](auto i) -> auto& { return big_data[i]; });

    auto random_idxs    = std::vector<uZ>(big_data_cnt * 4);
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
        auto preheat_span = big_data | stdv::take(preheat_cnt);
        auto data_view    = random_idxs | stdv::drop(preheat_cnt) | stdv::take(meas_cnt) | big_data_getter;
        auto avg_dur      = measure_inplace_iteration(runner, preheat_span, data_view);
        auto meas_dur     = avg_dur * meas_cnt;
        if (meas_dur < param.min_measurement) {
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

        auto t_factor = 2.6;    // Approximate for big n and 99.5 confidence
        auto delta    = t_factor * sd / std::sqrt(static_cast<f64>(durations.size()));
        stats         = {mean_dur, sd, delta};
    };
    auto filter_big = [&] {
        auto filtered_durations = std::vector<nano_s>();
        filtered_durations.reserve(durations.size());
        auto out = std::back_inserter(filtered_durations);
        stdr::copy_if(durations, out, [&](auto v) {
            return v.count() == v.count() && v.count() < stats.mean_dur.count() + stats.sd * 7;
        });
        using std::swap;
        swap(filtered_durations, durations);
    };


    durations.reserve(random_idxs.size() / meas_cnt * 4);
    uZ repeats = 1;
    while (true) {
        for (auto i: stdv::iota(repeats / 2, repeats)) {
            for (auto sample_idxs: random_idxs | stdv::chunk(meas_cnt + preheat_cnt)) {
                auto preheat_view = sample_idxs | stdv::take(preheat_cnt) | big_data_getter;
                auto data_view    = sample_idxs | stdv::drop(preheat_cnt) | big_data_getter;
                auto dur          = measure_inplace_iteration(runner, preheat_view, data_view);
                if (!std::isinf(dur.count()))
                    durations.emplace_back(dur);
            }
        }
        update_stats();
        if (stats.conf_delta / stats.mean_dur.count() < 0.01) {
            filter_big();
            update_stats();
            if (stats.conf_delta / stats.mean_dur.count() < 0.001)
                break;
        }
        repeats *= 2;
        if (repeats > 64000)
            break;
    }

    std::print("{} measurements of {} samples\n", durations.size(), meas_cnt);
    auto delta_perc = stats.conf_delta / stats.mean_dur.count() * 100;
    std::print("Mean: {}, standard deviation: {:.3}, t: {:.2}%\n", stats.mean_dur, stats.sd, delta_perc);
    auto [min, max] = stdr::minmax_element(durations);
    std::print("Min: {}, max: {}\n", *min, *max);
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

    uZ fft_size = 512;
    std::println("pcx:");
    measure_size<pcx_runner<f32>>(fft_size);
    std::println("kfr:");
    measure_size<kfr_runner<f32>>(fft_size);

    return 0;
}
