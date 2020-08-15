// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License

#include "../include/lcg.h"
#include "../include/version.h"
#include <iostream>

#define ASSERT(expr) if (!(expr)) std::cerr << "Assertion failed: " #expr " at line " __FILE__ ":" << __LINE__ << std::endl;

void test_lcg() {
    ASSERT(lcg::dfz2seed(47) == 0x2FC0CFC54419LL);
    ASSERT(lcg::seed2dfz(0x2FC0CFC54419LL) == 47);
}
void test_version(){
    ASSERT(version::MC_1_16.id==530);
}
void test_version_cmp(){
    ASSERT(version::MC_1_16>version::MC_1_15);
}
int main() {
    test_lcg();
    test_version();
    test_version_cmp();
    std::cout << "All tests have been run" << std::endl;
    return 0;
}
