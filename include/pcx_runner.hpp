#pragma once
#include "common.hpp"
#include "pcx/fft.hpp"

#include <concepts>

template<std::floating_point T>
class pcx_runner {
public:
    using real = T;

    explicit pcx_runner(uZ size)
    : m_plan(size) {
        // , m_tmp(size) {
        // m_permuter         = m_permuter.insert_indexes(m_idxs, size);
        // m_permuter.idx_ptr = m_idxs.data();
    };

    void fft(ccx_range_of<T> auto& data) const {
        m_plan.fft(data);
        // m_plan.fft(m_tmp, data);
        // auto src_data = pcx::detail_::sequential_data_info<T>{.data_ptr = reinterpret_cast<T*>(m_tmp.data())};
        // auto dst_data =
        //     pcx::detail_::sequential_data_info<T>{.data_ptr = reinterpret_cast<T*>(stdr::data(data))};
        // auto permuter = m_permuter;
        // stdr::copy(m_tmp, data.begin());
        // permuter.sequential_permute(pcx::cxpack<1, T>{}, pcx::cxpack<1, T>{}, dst_data, src_data);
    };
    void fft(ccx_range_of<T> auto& dest, const ccx_range_of<T> auto& src) const {
        m_plan.fft(dest, src);
    };

private:
    pcx::fft_plan<T, pcx::fft_options{.pt = pcx::fft_permutation::normal}> m_plan;
    // mutable std::vector<std::complex<T>>                                   m_tmp;
    // std::vector<u32>                                                       m_idxs;
    // pcx::detail_::br_permuter_sequential<16, false>                        m_permuter{};
};
