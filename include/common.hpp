#pragma once

#include <cstdint>
#include <ranges>

using f32 = float;
using f64 = double;

using uZ  = std::size_t;
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8  = uint8_t;

using iZ  = std::ptrdiff_t;
using i64 = int64_t;
using i32 = int32_t;
using i16 = int16_t;
using i8  = int8_t;

namespace stdr = std::ranges;
namespace stdv = std::views;

template<typename R, typename T>
concept ccx_range_of = stdr::contiguous_range<R> && std::same_as<stdr::range_value_t<R>, std::complex<T>>;
