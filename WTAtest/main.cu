#include "include/wta_kernel.hpp"
#include "include/common.hpp"
#include "include/utility.hpp"
#include "include/device_buffer.hpp"
using namespace srs;


template <size_t MAX_DISPARITY>
void test()
{
    std::cout << MAX_DISPARITY << std::endl;
}


int main(void) {

    std::cout << "wta test...." << std::endl;
    static constexpr size_t width = 1920, height = 1080, disparity = 64;
    static constexpr float uniqueness = 0.95f;
    const size_t pitch = width;
    const auto input = srs::generate_random_data<census_type>(
        width * height * disparity);
    const auto d_input = srs::to_device_vector(input);
    srs::DeviceBuffer<output_type> result = srs::DeviceBuffer<output_type>(pitch * height);
    srs::device_buffer wta_r(pitch * height * sizeof(ushort2));
    for (int i = 0; i < 1000; i++)
    {
        srs::enqueue_winner_takes_all(
            result.data(),
            d_input.data().get(),
            width,
            height,
            pitch,
            uniqueness,
            false,
            0);
        cudaStreamSynchronize(0);
    }
    const thrust::device_vector<output_type> d_actual(
        result.data(), result.data() + (pitch * height));
    const auto actual = to_host_vector(d_actual);
   /* std::cout << static_cast<uint64_t>(actual[0]) << std::endl;*/
    return 0;
}

