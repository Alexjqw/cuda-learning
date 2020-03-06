#include <iostream>
#include "include/test.hpp"
#include <vector>
int main()
{
    std::vector<int> a{ 1,2,4 };
    std::vector<int> b{ 1,2,4 };
    std::vector<int> c(b.size());
    add(a.data(), b.data(), c.data(),a.size());
    for (int i = 0; i < a.size(); i++)
    {
        std::cout << c[i] << std::endl;
    }
    std::cout << "success!!!" << std::endl;
}