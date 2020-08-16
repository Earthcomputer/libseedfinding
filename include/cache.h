// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#ifndef LIBSEEDFINDING_CACHE_H
#define LIBSEEDFINDING_CACHE_H
namespace cache{
    // make a int cache for size 16 32 48 64 80 96 112 128
    template<typename K,typename V> struct Cache{
        K map;
    };
}
#endif //LIBSEEDFINDING_CACHE_H
