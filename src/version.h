// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#ifndef SEEDFINDING_VERSION_H
#define SEEDFINDING_VERSION_H
#if __cplusplus < 201402L
#error version.h requires C++ 14
#endif
#include <string>

namespace version {
    enum Type {
        RELEASE,
        SNAPSHOT,
        OLD_ALPHA,
        OLD_BETA
    };
    struct Version {
        int id;
        Type type;
        std::string releaseTime;
    };

    inline bool operator==(const version::Version version1, const version::Version version2) {
        return version1.id == version2.id;
    }

    inline bool operator>(const version::Version version1, const version::Version version2) {
        return version1.id > version2.id;
    }

    inline bool operator<(const version::Version version1, const version::Version version2) {
        return version1.id < version2.id;
    }

    inline bool operator>=(const version::Version version1, const version::Version version2) {
        return version1.id >= version2.id;
    }

    inline bool operator<=(const version::Version version1, const version::Version version2) {
        return version1.id <= version2.id;
    }

    inline bool operator!=(const version::Version version1, const version::Version version2) {
        return version1.id != version2.id;
    }

    inline bool is_release(const version::Version version) {
        return version.type == RELEASE;
    }

    inline bool is_snapshot(const version::Version version) {
        return version.type == SNAPSHOT;
    }

    inline bool is_beta(const version::Version version) {
        return version.type == OLD_BETA;
    }

    inline bool is_alpha(const version::Version version) {
        return version.type == OLD_ALPHA;
    }

    /////////////////////////GENERATION////////////////////////////////


}
#endif //SEEDFINDING_VERSION_H
