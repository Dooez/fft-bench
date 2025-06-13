cmake_minimum_required(VERSION 3.18)
project(fft_bench LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(INSTALL_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/deps/install")

include(FetchContent)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

# FetchContent_Declare(
#     pcx
#     GIT_REPOSITORY git@github.com:Dooez/pcx.git
#     GIT_TAG main
# )
# FetchContent_MakeAvailable(pcx)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../pcx" "${CMAKE_CURRENT_BINARY_DIR}/build-pcx")

FetchContent_Declare(
    kfr
    GIT_REPOSITORY https://github.com/kfrlib/kfr
    GIT_TAG main
)
set(KFR_ARCH avx512)
set(KFR_ENABLE_MULTIARCH OFF)
FetchContent_MakeAvailable(kfr)


add_executable(bench
    src/pcx.cpp
    src/kfr.cpp
    src/fftw3.cpp
    src/run.cpp
)
target_include_directories(bench PUBLIC include)
target_link_libraries(bench
    pcx::pcx
    kfr
    kfr_dft
    PkgConfig::FFTW
)
set_target_properties(bench
    PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_compile_options(bench PUBLIC -march=native)
