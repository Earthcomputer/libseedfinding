// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#include "../src/lcg.h"
#include <iostream>

#define ASSERT(expr) if (!(expr)) std::cerr << "Assertion failed: " #expr " at line " __FILE__ ":" << __LINE__ << std::endl;

void test_lcg() {
    ASSERT(lcg::dfz2seed(47) == 0x2FC0CFC54419LL);
    ASSERT(lcg::seed2dfz(0x2FC0CFC54419LL) == 47);
}

int main() {
    test_lcg();

    std::cout << "All tests have been run" << std::endl;
    return 0;
}
