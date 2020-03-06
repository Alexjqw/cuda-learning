windows

## build:
$ cd libSGM
$ mkdir build
$ cd build
$ cmake .. -G"Visual Studio 15 2017 Win64"

$ cmake .. -G"Visual Studio 16 2019"


## test：
./test --gtest_filter=FourPathAggregation.FourPathAggregationTest
