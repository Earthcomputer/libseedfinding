// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#ifndef LIBSEEDFINDING_BIGINTEGER_H
#define LIBSEEDFINDING_BIGINTEGER_H
#define BIGINT BigInteger<uint16_t,64>
#include "../util.h"
#include <cinttypes>
#include <cstring>
namespace bigInteger{

    namespace internal{
        /*
         * According to IEEE we will provide an interface for infinite precision integer operation,
         * each integer will be split in T bits (usually 16 bits) words up to 2^16 (65536) ones, this result
         * in a total bit length of 1048576 bits or enough to store the universe (using size: 2^124) 8456 times.
         * We use template to have a Word size (W) as well as a type for each word, we recommend using uint16_t
         * as T and W to be less than 64 (which gives 16*64=1024bits)
         */
        template <typename T,uint16_t W>
        struct BigInteger{
            bool sign;
            T words[W];
            uint16_t words_size;
        };

        BigInteger(std::string str){

        }
    }
}
#endif //LIBSEEDFINDING_BIGINTEGER_H
