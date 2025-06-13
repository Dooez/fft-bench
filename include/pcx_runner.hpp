#pragma once
#include "common.hpp"
#include "pcx/fft.hpp"

#include <concepts>

template<std::floating_point T>
class pcx_runner {
public:
    using real = T;

    explicit pcx_runner(uZ size)
    : m_plan(size) {};

    void fft(ccx_range_of<T> auto& data) const {
        m_plan.fft(data);
    };
    void fft(ccx_range_of<T> auto& dest, const ccx_range_of<T> auto& src) const {
        m_plan.fft(dest, src);
    };

private:
    pcx::fft_plan<T, pcx::fft_options{.pt = pcx::fft_permutation::normal}> m_plan;
};
