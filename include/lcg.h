// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#ifndef LIBSEEDFINDING_LCG_H
#define LIBSEEDFINDING_LCG_H

#include <cinttypes>
#include <limits>
#include <type_traits>
#include "util.h"

#if CPP_VER < 201402L
#error lcg.h requires C++ 14
#endif


/**
 * Contains an implementation of the Java LCG and utility functions surrounding it.
 * Many functions here are constexpr, meaning that they can be evaluated at compile-time if their inputs are
 * also statically known. Otherwise, they are usually inlined by the compiler instead.
 */
namespace lcg {
    typedef uint64_t Random;

    const uint64_t MULTIPLIER = 0x5deece66dLL;
    const uint64_t ADDEND = 0xbLL;
    const uint64_t MASK = 0xffffffffffffL;

    /// The minimum distance that two nextFloat values can be from each other. May be used to iterate over all valid floats.
    static DEVICEABLE_CONST float FLOAT_UNIT = 1.0f / static_cast<float>(1L << 24);
    /// The minimum distance that two nextDouble values can be from each other. May be used to iterate over all valid doubles.
    static DEVICEABLE_CONST double DOUBLE_UNIT = 1.0 / static_cast<double>(1LL << 53);

    // Declared here for forward reference
    template<int B>
    DEVICEABLE constexpr typename std::enable_if_t<(0 <= B && B <= 32), int32_t> next(Random &rand);

    /**
     * Contains internal functions. These are unstable, do not use them for any reason!
     * If you think you need something from in here, first look for alternatives, else consider adding something to the public API for it.
     * The internal functions are included in the header file so that the compiler can optimize using their implementations.
     * Many of these functions are force-inlined. This is to ensure the public force-inlined functions are fully inlined properly.
     */
    namespace internal {
        // for returning multiple values
        struct LCG {
            uint64_t multiplier;
            uint64_t addend;
        };

        DEVICEABLE FORCEINLINE constexpr LCG combine(uint64_t calls) {
            uint64_t multiplier = 1;
            uint64_t addend = 0;

            uint64_t intermediate_multiplier = MULTIPLIER;
            uint64_t intermediate_addend = ADDEND;

            for (uint64_t k = calls; k != 0; k >>= 1) {
                if ((k & 1) != 0) {
                    multiplier *= intermediate_multiplier;
                    addend = intermediate_multiplier * addend + intermediate_addend;
                }

                intermediate_addend = (intermediate_multiplier + 1) * intermediate_addend;
                intermediate_multiplier *= intermediate_multiplier;
            }

            multiplier &= MASK;
            addend &= MASK;

            return {multiplier, addend};
        }

        DEVICEABLE FORCEINLINE constexpr LCG combine(int64_t calls) {
            return combine(static_cast<uint64_t>(calls));
        }

        DEVICEABLE FORCEINLINE constexpr uint64_t gcd(uint64_t a, uint64_t b) {
            if (b == 0) {
                return a;
            }
            while (true) {
                a %= b;
                if (a == 0) {
                    return b;
                }
                b %= a;
                if (b == 0) {
                    return a;
                }
            }
        }

        // Returns modulo inverse of a with
        // respect to m using extended Euclid
        // Algorithm Assumption: a and m are
        // coprimes, i.e., gcd(a, m) = 1
        // stolen code, now handles the case where gcd(a,m) != 1
        DEVICEABLE FORCEINLINE constexpr uint64_t euclidean_helper(uint64_t a, uint64_t m) {
            uint64_t y = 0, x = 1;
            if (m == 1) {
                return 0;
            }
            uint64_t gcd_ = gcd(a, m);
            while (a > gcd_) {
                uint64_t q = a / m;
                uint64_t t = m;
                m = a % m;
                a = t;
                t = y;
                y = x - q * y;
                x = t;
            }
            return x;
        }

        DEVICEABLE FORCEINLINE constexpr uint64_t theta(uint64_t num) {
            if (num % 4 == 3) {
                num = (1LL << 50) - num;
            }

            // xhat = num
            uint64_t xhat_lo = num;
            uint64_t xhat_hi = 0;

            // raise xhat to the power of 2^49 by squaring it 49 times
            for (int i = 0; i < 49; i++) {
                // https://www.codeproject.com/Tips/618570/UInt-Multiplication-Squaring
                uint64_t r1 = xhat_lo & 0xffffffffLL;
                uint64_t t = r1 * r1;
                uint64_t w3 = t & 0xffffffffLL;
                uint64_t k = t >> 32;
                uint64_t r = xhat_lo >> 32;
                uint64_t m = r * r1;
                t = m + k;
                uint64_t w2 = t & 0xffffffffLL;
                uint64_t w1 = t >> 32;
                t = m + w2;
                k = t >> 32;
                uint64_t new_hi = r * r + w1 + k;
                uint64_t new_lo = (t << 32) + w3;
                new_hi += (xhat_hi * xhat_lo) << 1;
                xhat_lo = new_lo;
                xhat_hi = new_hi;
            }
            xhat_hi &= (1LL << (99 - 64)) - 1;

            // xhat--
            if (xhat_lo == 0) xhat_hi--;
            xhat_lo--;

            // xhat >>= 51
            xhat_lo = (xhat_lo >> 51) | (xhat_hi << (64 - 51));

            // xhat &= MASK
            xhat_lo &= MASK;
            return xhat_lo;
        }

        DEVICEABLE constexpr int32_t dynamic_next_int_power_of_2(Random &rand, int32_t n) {
            return static_cast<int32_t>((static_cast<uint64_t>(next<31>(rand)) * static_cast<uint64_t>(n)) >> 31);
        }
    }


    /// Unsigned equivalent of combined_lcg<N>.
    template<uint64_t N>
    struct ucombined_lcg {
        /// The multiplier of the LCG that advances by N.
        static const uint64_t multiplier = internal::combine(N).multiplier;
        /// The addend of the LCG that advances by N.
        static const uint64_t addend = internal::combine(N).addend;
    };

    /**
     * Contains the multiplier and addend of the LCG equivalent to advancing the Java LCG by N.
     * N may be negative to signal a backwards advance.
     */
    template<int64_t N>
    struct combined_lcg {
        /// The multiplier of the LCG that advances by N.
        static const uint64_t multiplier = ucombined_lcg<static_cast<uint64_t>(N)>::multiplier;
        /// The addend of the LCG that advances by N.
        static const uint64_t addend = ucombined_lcg<static_cast<uint64_t>(N)>::addend;
    };

    /// Advances the Random by an unsigned N steps, which defaults to 1. Runs in O(1) time because of compile-time optimizations.
    template<uint64_t N = 1>
    DEVICEABLE constexpr void uadvance(Random &rand) {
        rand = (rand * ucombined_lcg<N>::multiplier + ucombined_lcg<N>::addend) & MASK;
    }

    /**
     * Advances the Random by N steps, which defaults to 1. Runs in O(1) time because of compile-time optimizations.
     * N may be negative to signal a backwards advance.
     */
    template<int64_t N = 1>
    DEVICEABLE constexpr void advance(Random &rand) {
        uadvance<static_cast<uint64_t>(N)>(rand);
    }

    /// Force-inlined version of dynamic_advance. Do not use unless profiling tells you that the compiler is not inlining anyway!
    DEVICEABLE FORCEINLINE constexpr void dynamic_advance_inline(Random &rand, uint64_t n) {
#define ADVANCE_BIT(N) if (n < (1LL << N)) return;\
           if (n & (1LL << N)) uadvance<1LL << N>(rand);
        ADVANCE_BIT(0)
        ADVANCE_BIT(1)
        ADVANCE_BIT(2)
        ADVANCE_BIT(3)
        ADVANCE_BIT(4)
        ADVANCE_BIT(5)
        ADVANCE_BIT(6)
        ADVANCE_BIT(7)
        ADVANCE_BIT(8)
        ADVANCE_BIT(9)
        ADVANCE_BIT(10)
        ADVANCE_BIT(11)
        ADVANCE_BIT(12)
        ADVANCE_BIT(13)
        ADVANCE_BIT(14)
        ADVANCE_BIT(15)
        ADVANCE_BIT(16)
        ADVANCE_BIT(17)
        ADVANCE_BIT(18)
        ADVANCE_BIT(19)
        ADVANCE_BIT(20)
        ADVANCE_BIT(21)
        ADVANCE_BIT(22)
        ADVANCE_BIT(23)
        ADVANCE_BIT(24)
        ADVANCE_BIT(25)
        ADVANCE_BIT(26)
        ADVANCE_BIT(27)
        ADVANCE_BIT(28)
        ADVANCE_BIT(29)
        ADVANCE_BIT(30)
        ADVANCE_BIT(31)
        ADVANCE_BIT(32)
        ADVANCE_BIT(33)
        ADVANCE_BIT(34)
        ADVANCE_BIT(35)
        ADVANCE_BIT(36)
        ADVANCE_BIT(37)
        ADVANCE_BIT(38)
        ADVANCE_BIT(39)
        ADVANCE_BIT(40)
        ADVANCE_BIT(41)
        ADVANCE_BIT(42)
        ADVANCE_BIT(43)
        ADVANCE_BIT(44)
        ADVANCE_BIT(45)
        ADVANCE_BIT(46)
        ADVANCE_BIT(47)
#undef ADVANCE_BIT
    }

    /// Advances the Random by an unsigned n steps. Used when n is not known at compile-time. Runs in O(log(n)) time.
    DEVICEABLE constexpr void dynamic_advance(Random &rand, uint64_t n) {
        dynamic_advance_inline(rand, n);
    }

    /// Force-inlined version of dynamic_advance. Do not use unless profiling tells you that the compiler is not inlining anyway!
    DEVICEABLE FORCEINLINE constexpr void dynamic_advance_inline(Random &rand, int64_t n) {
        dynamic_advance_inline(rand, static_cast<uint64_t>(n));
    }

    /**
     * Advances the Random by n steps. Used when n is not known at compile-time. Runs in O(log(n)) time.
     * n may be negative to signal a backwards advance.
     */
    DEVICEABLE constexpr void dynamic_advance(Random &rand, int64_t n) {
        dynamic_advance_inline(rand, n);
    }

    /// Force-inlined version of dfz2seed. Do not use unless profiling tells you that the compiler is not inlining anyway!
    DEVICEABLE FORCEINLINE constexpr Random dfz2seed_inline(uint64_t dfz) {
        Random seed = 0;
        dynamic_advance_inline(seed, dfz);
        return seed;
    }

    /**
     * Converts a Distance From Zero (DFZ) value to a Random seed.
     * DFZ is a representation of a seed which is the number of LCG calls required to get from seed 0 to that seed.
     * To get Random outputs from a DFZ value it must first be converted to a seed, which is done in O(log(dfz)).
     * In various situations, especially far GPU parallelization, it may be useful to represent seeds this way.
     */
    DEVICEABLE constexpr Random dfz2seed(uint64_t dfz) {
        return dfz2seed_inline(dfz);
    }

    /// Force-inlined version of seed2dfz. Do not use unless profiling tells you that the compiler is not inlining anyway!
    DEVICEABLE FORCEINLINE constexpr uint64_t seed2dfz_inline(Random seed) {
        uint64_t a = 25214903917LL;
        uint64_t b = (((seed * (MULTIPLIER - 1)) * 179120439724963LL) + 1) & ((1LL << 50) - 1);
        uint64_t abar = internal::theta(a);
        uint64_t bbar = internal::theta(b);
        uint64_t gcd_ = internal::gcd(abar, (1LL << 48));
        return (bbar * internal::euclidean_helper(abar, (1LL << 48)) & 0x3FFFFFFFFFFFLL) / gcd_; //+ i*(1L << 48)/gcd;
    }

    /**
     * Converts a Random seed to DFZ form. See dfz2seed for a description of DFZ form.
     * This function should be called reservedly, as although it is O(1), it is relatively slow.
     */
    DEVICEABLE constexpr uint64_t seed2dfz(Random seed) {
        return seed2dfz_inline(seed);
    }

    /// Advances the LCG and gets the upper B bits from it.
    template<int B>
    DEVICEABLE constexpr typename std::enable_if_t<(0 <= B && B <= 32), int32_t> next(Random &rand) {
        advance(rand);
        return static_cast<int32_t>(rand >> (48 - B));
    }

    /// Does an unbounded nextInt call and returns the result.
    DEVICEABLE constexpr int32_t next_int_unbounded(Random &rand) {
        return next<32>(rand);
    }

    /// Does a bounded nextInt call with bound N.
    template<int32_t N>
    DEVICEABLE constexpr typename std::enable_if_t<(N > 0) && ((N & -N) == N), int32_t> next_int(Random &rand) {
        return static_cast<int32_t>((static_cast<uint64_t>(next<31>(rand)) * static_cast<uint64_t>(N)) >> 31);
    }

    template<int32_t N>
    DEVICEABLE constexpr typename std::enable_if_t<(N > 0) && ((N & -N) != N), int32_t> next_int(Random &rand) {
        int32_t bits = next<31>(rand);
        int32_t val = bits % N;
        while (bits - val + (N - 1) < 0) {
            bits = next<31>(rand);
            val = bits % N;
        }
        return val;
    }

    /**
     * Does a bounded nextInt call with bound N. If N is not a power of 2, then it makes the assumption that the loop
     * does not iterate more than once. The probability of this being correct depends on N, but for small N this
     * function is extremely likely to have the same effect as next_int.
     */
    template<int32_t N>
    DEVICEABLE constexpr typename std::enable_if_t<(N > 0), int32_t> next_int_fast(Random &rand) {
        if ((N & -N) == N) {
            return next_int<N>(rand);
        } else {
            return next<31>(rand) % N;
        }
    }

    /// Does a bounded nextInt call with bound n, used when n is not known in advance.
    DEVICEABLE constexpr int32_t dynamic_next_int(Random &rand, int32_t n) {
        if ((n & -n) == n) {
            return internal::dynamic_next_int_power_of_2(rand, n);
        } else {
            int32_t bits = next<31>(rand);
            int32_t val = bits % n;
            while (bits - val + (n - 1) < 0) {
                bits = next<31>(rand);
                val = bits % n;
            }
            return val;
        }
    }

    /**
     * Does a bounded nextInt call with bound n, using the "fast" approach, used when n is not known in advance.
     * See next_int_fast for a description of the fast approach.
     */
    DEVICEABLE constexpr int32_t dynamic_next_int_fast(Random &rand, int32_t n) {
        if ((n & -n) == n) {
            return internal::dynamic_next_int_power_of_2(rand, n);
        } else {
            return next<31>(rand) % n;
        }
    }

    /// Does a nextLong call.
    DEVICEABLE constexpr int64_t next_long(Random &rand) {
        // separate out calls due to unspecified evaluation order in C++
        int32_t hi = next<32>(rand);
        int32_t lo = next<32>(rand);
        return (static_cast<int64_t>(hi) << 32) + static_cast<int64_t>(lo);
    }

    /// Does an unsigned nextLong call.
    DEVICEABLE constexpr uint64_t next_ulong(Random &rand) {
        return static_cast<uint64_t>(next_long(rand));
    }

    /// Does a nextBoolean call.
    DEVICEABLE constexpr bool next_bool(Random &rand) {
        return next<1>(rand) != 0;
    }

    /// Does a nextFloat call.
    DEVICEABLE std::enable_if_t<std::numeric_limits<float>::is_iec559, float> next_float(Random &rand) {
        return static_cast<float>(next<24>(rand)) * FLOAT_UNIT;
    }

    /// Does a nextDouble call.
    DEVICEABLE std::enable_if_t<std::numeric_limits<double>::is_iec559, double> next_double(Random &rand) {
        // separate out calls due to unspecified evaluation order in C++
        int32_t hi = next<26>(rand);
        int32_t lo = next<27>(rand);
        return static_cast<double>((static_cast<int64_t>(hi) << 27) + static_cast<int64_t>(lo)) * DOUBLE_UNIT;
    }
}

#endif //LIBSEEDFINDING_LCG_H
