// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#ifndef LIBSEEDFINDING_KERNELS_H
#define LIBSEEDFINDING_KERNELS_H

#ifndef __CUDACC__
#error "kernels.h can only be used with CUDA"
#endif

#include <functional>
#include "lcg.h"

namespace kernels {
    typedef bool(*seed_tester)(lcg::Random);

    enum class seed_format {
        SEED, DFZ
    };

    template<seed_tester tester, seed_format input_format = seed_format::SEED, seed_format output_format = input_format, typename count_t = uint32_t>
    __global__ void bruteforce(uint64_t offset, count_t* count, uint64_t* buffer) {
        uint64_t value = offset + blockIdx.x * blockDim.x + threadIdx.x;
        lcg::Random seed = input_format == seed_format::SEED ? value : lcg::dfz2seed_inline(value);

        if (tester(seed)) {
            count_t index = atomicAdd(count, 1);
            if (output_format == seed_format::SEED) {
                buffer[index] = seed;
            } else if (input_format == seed_format::DFZ) {
                buffer[index] = value;
            } else {
                buffer[index] = lcg::seed2dfz_inline(seed);
            }
        }
    }

    template<seed_tester tester, seed_format input_format = seed_format::SEED, seed_format output_format = input_format, typename count_t = uint32_t>
    __global__ void filter(count_t count, uint64_t* bothput) {
        count_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > count) {
            return;
        }
        uint64_t value = bothput[index];
        if (value == 0) {
            return;
        }
        lcg::Random seed = input_format == seed_format::SEED ? value : lcg::dfz2seed_inline(value);

        if (tester(seed)) {
            if (input_format == seed_format::SEED && output_format == seed_format::DFZ) {
                bothput[index] = lcg::seed2dfz_inline(seed);
            } else if (input_format == seed_format::DFZ && output_format == seed_format::SEED) {
                bothput[index] = seed;
            }
        } else {
            bothput[index] = 0;
        }
    }

    template<typename input_count_t = uint32_t, typename output_count_t = input_count_t>
    __global__ void compact(input_count_t input_count, uint64_t* input, output_count_t* output_count, uint64_t* output) {
        input_count_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > input_count) {
            return;
        }
        uint64_t value = input[index];
        if (value != 0) {
            output_count_t output_index = atomicAdd(output_count, 1);
            output[output_index] = value;
        }
    }
}

#endif //LIBSEEDFINDING_KERNELS_H
