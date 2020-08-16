// Copyright (c) The Minecraft Seed Finding Team
// version 1.0.1
// MIT License

#ifndef LIBSEEDFINDING_UTIL_H
#define LIBSEEDFINDING_UTIL_H

#ifdef _MSVC_LANG
#define CPP_VER _MSVC_LANG
#else
#define CPP_VER __cplusplus
#endif

#if defined(__CUDACC__)
#define FORCEINLINE __forceinline__
#elif defined(__GNUC__)
#define FORCEINLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define FORCEINLINE __forceinline
#else
#warning Using unknown compiler, various optimizations may not be enabled
#define FORCEINLINE
#endif

#ifdef __CUDACC__
#define DEVICEABLE __host__ __device__
#else
#define DEVICEABLE
#endif

#ifdef __CUDA_ARCH__
#define DEVICE_FORCEINLINE FORCEINLINE
#define HOST_FORCEINLINE
#define DEVICEABLE_CONST __constant__ __device__
#else
#define DEVICE_FORCEINLINE
#define HOST_FORCEINLINE FORCEINLINE
#define DEVICEABLE_CONST const
#endif

#endif //LIBSEEDFINDING_UTIL_H
