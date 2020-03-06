#ifndef SGM_W_T_A_HPP
#define SGM_W_T_A_HPP

#include"../include/utility.hpp"

namespace srs{



void enqueue_winner_takes_all(
    output_type *left_dest,
    const census_type *src,
    int width,
    int height,
    int pitch,
 
    float uniqueness,
    bool subpixel,
    cudaStream_t stream);
}



#endif

