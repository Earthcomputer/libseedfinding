// Copyright (c) The Minecraft Seed Finding Team
//
// MIT License


#include <iostream>
#include <lcg.h>
#include <version.h>


int main() {
    std::cout << lcg::dfz2seed(47) << std::endl;
    std::cout << version::MC_1_16.id << std::endl;
    return 0;
}
