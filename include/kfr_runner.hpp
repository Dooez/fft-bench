#pragma once
#include "common.hpp"
#include "kfr/dft.hpp"

#include <concepts>

template<std::floating_point T>
class kfr_runner {
public:
    using real = T;

    explicit kfr_runner(uZ size)
    : m_plan(size, kfr::dft_order::internal)
    , m_temp(m_plan.temp_size) {};

    void fft(ccx_range_of<T> auto& data) const {
        m_plan.execute(stdr::data(data), stdr::data(data), m_temp.data());
    };
    void fft(ccx_range_of<T> auto& dest, const ccx_range_of<T> auto& src) const {
        m_plan.execute(stdr::data(dest), stdr::data(src), m_temp.data());
    };

private:
    kfr::dft_plan<real>     m_plan;
    mutable std::vector<u8> m_temp;
};
