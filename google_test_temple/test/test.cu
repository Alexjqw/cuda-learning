#include <gtest/gtest.h>
#include <utility>
#include <algorithm>
#include "../include/test.hpp"
namespace
{
    void test_add(const int* a, const int* b, int* c, const unsigned int size)
    {
        for (int i = 0; i < size; i++)
        {
            c[i] = a[i] + b[i];
        }
    }
}
TEST(TEST, UU)
{
    std::vector<int> a{ 1,2,4 };
    std::vector<int> b{ 1,2,4 };
    std::vector<int> c(b.size());
    std::vector<int> expect(b.size());
    add(a.data(), b.data(), c.data(), a.size());
    test_add(a.data(), b.data(), expect.data(), a.size());
    EXPECT_EQ(c, expect);
    //debug_compare(c.data(), expect.data(), pitch, height, 1);
	std::cout<<"hello"<<std::endl;
}
