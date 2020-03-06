#include"../include/test.hpp"

void add(const int* a, const int* b, int* c, const unsigned int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }

}