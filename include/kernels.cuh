// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#ifndef LIBSEEDFINDING_KERNELS_H
#define LIBSEEDFINDING_KERNELS_H

#ifndef __CUDACC__
#error "kernels.cuh can only be used with CUDA"
#endif

#include <functional>
#include "lcg.h"

/**
 * Boilerplate code for different types of kernels.
 */
namespace kernels {
    /// A function which takes a seed and returns a true or false value.
    typedef bool(*seed_predicate)(lcg::Random);

    /// A format for a seed to be in.
    enum class seed_format {
        /// Specifies that the value is equal to the seed of the Random.
        SEED,
        /// Specifies that the value is equal to the DFZ form of the seed of the Random. See lcg::dfz2seed for details.
        DFZ
    };

    /**
     * This kernel can be used for the first step in most brute forces. It runs over the entire seed-space and adds to
     * the output buffer where the given predicate matches. A single call to this kernel will only do a fraction of the
     * seed-space, to search the full seed-space, this function must be called in a loop with different offset values.
     *
     * @tparam PREDICATE The function used to test_cuda whether a seed should be added to the buffer. This function will
     *                      always receive seeds in seed_format::SEED form, regardless of the value of INPUT_ORDER.
     *                      Must be a __device__ function.
     * @tparam INPUT_ORDER The order in which input seeds are brute-forced. Defaults to seed_format::SEED, which is the
     *                      most efficient in most cases, however some optimized functions rely on seed_format::DFZ.
     *                      These functions state this in their documentation.
     * @tparam OUTPUT_FORMAT The format of seeds added to the output buffer. Defaults to seed_format::SEED (regardless
     *                      of input order).
     * @tparam count_t The type of the output size of the buffer. Defaults to uint32_t.
     * @param offset The start point of the bruteforce. This kernel will bruteforce gridDim * blockDim seeds starting
     *                  from this offset.
     * @param count The kernel will start adding seeds to buffer at this index. After the kernel has finished, this
     *              value will store the index of the next value to be inserted into buffer. That is, if count starts
     *              off as 0, then count will end up as the number of matching seeds added to the buffer.
     * @param buffer The buffer to add matching seeds to.
     */
    template<seed_predicate PREDICATE, seed_format INPUT_ORDER = seed_format::SEED, seed_format OUTPUT_FORMAT = seed_format::SEED, typename count_t = uint32_t>
    __global__ void bruteforce(uint64_t offset, count_t* count, uint64_t* buffer) {
        uint64_t value = offset + blockIdx.x * blockDim.x + threadIdx.x;
        lcg::Random seed = INPUT_ORDER == seed_format::SEED ? value : lcg::dfz2seed_inline(value);

        if (PREDICATE(seed)) {
            count_t index = atomicAdd(count, 1);
            if (OUTPUT_FORMAT == seed_format::SEED) {
                buffer[index] = seed;
            } else if (INPUT_ORDER == seed_format::DFZ) {
                buffer[index] = value;
            } else {
                buffer[index] = lcg::seed2dfz_inline(seed);
            }
        }
    }

    /**
     * This kernel can be used as a secondary step to filter down a list of seeds. Seeds are read from bothput, tested,
     * and if the predicate returns false, 0 is written to bothput. If INPUT_FORMAT is different from OUTPUT_FORMAT, the
     * seeds in bothput are also converted and written back. Note that this kernel assumes that seed 0 does not match.
     * If this is a concern, seed 0 should be tested separately on the host.
     *
     * @tparam PREDICATE The function used to test_cuda whether a seed should be added to the buffer. This function will
     *                      always receive seeds in seed_format::SEED form, regardless of the value of INPUT_ORDER.
     *                      Must be a __device__ function.
     * @tparam INPUT_FORMAT The format of the input seeds in bothput. If not equal to seed_format::SEED, seeds in
     *                      bothput will be converted to seed_format::SEED before being passed to PREDICATE.
     * @tparam OUTPUT_FORMAT The format of the seeds in bothput afterwards. Defaults to seed_format::SEED.
     * @tparam count_t The type of count.
     * @param count The number of seeds in bothput.
     * @param bothput The seed buffer.
     */
    template<seed_predicate PREDICATE, seed_format INPUT_FORMAT = seed_format::SEED, seed_format OUTPUT_FORMAT = seed_format::SEED, typename count_t = uint32_t>
    __global__ void filter(count_t count, uint64_t* bothput) {
        count_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > count) {
            return;
        }
        uint64_t value = bothput[index];
        if (value == 0) {
            return;
        }
        lcg::Random seed = INPUT_FORMAT == seed_format::SEED ? value : lcg::dfz2seed_inline(value);

        if (PREDICATE(seed)) {
            if (INPUT_FORMAT == seed_format::SEED && OUTPUT_FORMAT == seed_format::DFZ) {
                bothput[index] = lcg::seed2dfz_inline(seed);
            } else if (INPUT_FORMAT == seed_format::DFZ && OUTPUT_FORMAT == seed_format::SEED) {
                bothput[index] = seed;
            }
        } else {
            bothput[index] = 0;
        }
    }

    /**
     * Copies seeds from input into output, skipping 0 values. This should only be used if the input buffer is both
     * large and sparse. In many cases, a simple loop on the host is more efficient due to the overhead of copying
     * memory to and from the device.
     *
     * @tparam input_count_t The type of input_count.
     * @tparam output_count_t The type of output_count.
     * @param input_count The number of seeds in the input buffer.
     * @param input The input buffer.
     * @param output_count The number of seeds in the output buffer will be written to this value. Should be initialized
     *                      to 0 before calling this kernel.
     * @param output The output buffer.
     */
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
