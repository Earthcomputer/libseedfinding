// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#ifndef SEEDFINDING_SIMPLEX_H
#define SEEDFINDING_SIMPLEX_H

#include "lcg.h"
#include "cache.h"
// constant definition for simplex
#define F2 0.3660254037844386
#define G2 0.21132486540518713
#define F3 0.3333333333333333
#define G3 0.16666666666666666

namespace noise {
    struct Noise {
        double xo;
        double yo;
        double zo;
        uint8_t permutations[256];
    };

    static inline void initOctaves(Noise octaves[], lcg::Random random, int nbOctaves) {
        for (int i = 0; i < nbOctaves; ++i) {
            octaves[i].xo = lcg::next_double(random) * 256.0;
            octaves[i].yo = lcg::next_double(random) * 256.0;
            octaves[i].zo = lcg::next_double(random) * 256.0;
            uint8_t *permutations = octaves[i].permutations;
            uint8_t j = 0;
            do {
                permutations[j] = j;
            } while (j++ != 255);
            uint8_t index = 0;
            do {
                uint32_t randomIndex = lcg::dynamic_next_int(random, 256u - index) + index;
                if (randomIndex != index) {
                    // swap
                    permutations[index] ^= permutations[randomIndex];
                    permutations[randomIndex] ^= permutations[index];
                    permutations[index] ^= permutations[randomIndex];
                }
            } while (index++ != 255);
        }
    }
}
namespace simplex {
    struct Simplex{
        noise::Noise noise;
        cache::Cache<
    }
    DEVICEABLE_CONST int grad2[12][2] = {{1,  1,},
                              {-1, 1,},
                              {1,  -1,},
                              {-1, -1,},
                              {1,  0,},
                              {-1, 0,},
                              {1,  0,},
                              {-1, 0,},
                              {0,  1,},
                              {0,  -1,},
                              {0,  1,},
                              {0,  -1,}};

    static inline void simplexNoise(double **buffer, double chunkX, double chunkZ, int x, int z, double offsetX, double offsetZ, double octaveFactor, PermutationTable permutationTable) {
        int k = 0;
        uint8_t *permutations = permutationTable.permutations;
        for (int X = 0; X < x; X++) {
            double XCoords = (chunkX + (double) X) * offsetX + permutationTable.xo;
            for (int Z = 0; Z < z; Z++) {
                double ZCoords = (chunkZ + (double) Z) * offsetZ + permutationTable.yo;
                // Skew the input space to determine which simplex cell we're in
                double hairyFactor = (XCoords + ZCoords) * F2;
                auto tempX = static_cast<int32_t>(XCoords + hairyFactor);
                auto tempZ = static_cast<int32_t>(ZCoords + hairyFactor);
                int32_t xHairy = (XCoords + hairyFactor < tempX) ? (tempX - 1) : (tempX);
                int32_t zHairy = (ZCoords + hairyFactor < tempZ) ? (tempZ - 1) : (tempZ);
                double d11 = (double) (xHairy + zHairy) * G2;
                double X0 = (double) xHairy - d11; // Unskew the cell origin back to (x,y) space
                double Y0 = (double) zHairy - d11;
                double x0 = XCoords - X0; // The x,y distances from the cell origin
                double y0 = ZCoords - Y0;
                // For the 2D case, the simplex shape is an equilateral triangle.
                // Determine which simplex we are in.
                int offsetSecondCornerX, offsetSecondCornerZ; // Offsets for second (middle) corner of simplex in (i,j) coords

                if (x0 > y0) {  // lower triangle, XY order: (0,0)->(1,0)->(1,1)
                    offsetSecondCornerX = 1;
                    offsetSecondCornerZ = 0;
                } else { // upper triangle, YX order: (0,0)->(0,1)->(1,1)
                    offsetSecondCornerX = 0;
                    offsetSecondCornerZ = 1;
                }

                double x1 = (x0 - (double) offsetSecondCornerX) + G2; // Offsets for middle corner in (x,y) unskewed coords
                double y1 = (y0 - (double) offsetSecondCornerZ) + G2;
                double x2 = (x0 - 1.0) + 2.0 * G2; // Offsets for last corner in (x,y) unskewed coords
                double y2 = (y0 - 1.0) + 2.0 * G2;

                // Work out the hashed gradient indices of the three simplex corners
                uint32_t ii = (uint32_t) xHairy & 0xffu;
                uint32_t jj = (uint32_t) zHairy & 0xffu;
                uint8_t gi0 = permutations[ii + permutations[jj]] % 12u;
                uint8_t gi1 = permutations[ii + offsetSecondCornerX + permutations[jj + offsetSecondCornerZ]] % 12u;
                uint8_t gi2 = permutations[ii + 1 + permutations[jj + 1]] % 12u;

                // Calculate the contribution from the three corners
                double t0 = 0.5 - x0 * x0 - y0 * y0;
                double n0;
                if (t0 < 0.0) {
                    n0 = 0.0;
                } else {
                    t0 *= t0;
                    n0 = t0 * t0 * ((double) grad2[gi0][0] * x0 + (double) grad2[gi0][1] * y0);  // (x,y) of grad2 used for 2D gradient
                }
                double t1 = 0.5 - x1 * x1 - y1 * y1;
                double n1;
                if (t1 < 0.0) {
                    n1 = 0.0;
                } else {
                    t1 *= t1;
                    n1 = t1 * t1 * ((double) grad2[gi1][0] * x1 + (double) grad2[gi1][1] * y1);
                }
                double t2 = 0.5 - x2 * x2 - y2 * y2;
                double n2;
                if (t2 < 0.0) {
                    n2 = 0.0;
                } else {
                    t2 *= t2;
                    n2 = t2 * t2 * ((double) grad2[gi2][0] * x2 + (double) grad2[gi2][1] * y2);
                }
                // Add contributions from each corner to get the final noise value.
                // The result is scaled to return values in the interval [-1,1].
                (*buffer)[k] = (*buffer)[k] + 70.0 * (n0 + n1 + n2) * octaveFactor;
                k++;

            }

        }
    }
}
#endif //SEEDFINDING_SIMPLEX_H
