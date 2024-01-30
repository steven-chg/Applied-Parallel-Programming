#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

/* Sweeping Parameters Optimization */
#define TILE_WIDTH 20                   // Baseline TILE_WIDTH 16; Tested 18, 19, and 20 (Optimal)

/* Constant Memory for Weight Matrix Optimization */
__constant__ float weight_matrix[16000];                // initialize to a large size

/* Tuning with Restrict and Loop Unrolling Optimization */
// Restrict Input, Output, and Mask Pointers 
__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;      
    const int W_out = (W - K)/S + 1;      

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) weight_matrix[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]        // change to access weight matrix in constant memory

    // Insert your GPU convolution kernel code here
    int W_grid = (W_out - 1)/TILE_WIDTH + 1;      // # of horizontal tiles per output map

    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;
    if(h < H_out && w < W_out){
        for(int c = 0; c < C; c++){

            /* Tuning with Restrict and Loop Unrolling Optimization */
            if(K == 1){

                /* Tuning with Unroll K = 1 */     
                // loop p = 0, q = 0
                acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);

            } else if(K == 2){

                /* Tuning with Unroll K = 2 */     
                // loop p = 0, q = 0~1
                acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);

                // loop p = 1, q = 0~1
                acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);

            } else if(K == 3){                                                      // TESTED UNROLL IF K <= 3

                /* Tuning with Unroll K = 3 */     
                // loop p = 0, q = 0~2
                acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);

                // loop p = 1, q = 0~2
                acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);

                // loop p = 2, q = 0~2
                acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);

            } else if(K == 4){

                // /* Tuning with Unroll K = 4 */
                // loop p = 0, q = 0~3
                acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);
                acc += in_4d(b, c, S*h+0, S*w+3) * mask_4d(m, c, 0, 3);

                // loop p = 1, q = 0~3
                acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);
                acc += in_4d(b, c, S*h+1, S*w+3) * mask_4d(m, c, 1, 3);

                // loop p = 2, q = 0~3
                acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);
                acc += in_4d(b, c, S*h+2, S*w+3) * mask_4d(m, c, 2, 3);

                // loop p = 3, q = 0~3
                acc += in_4d(b, c, S*h+3, S*w+0) * mask_4d(m, c, 3, 0);
                acc += in_4d(b, c, S*h+3, S*w+1) * mask_4d(m, c, 3, 1);
                acc += in_4d(b, c, S*h+3, S*w+2) * mask_4d(m, c, 3, 2);
                acc += in_4d(b, c, S*h+3, S*w+3) * mask_4d(m, c, 3, 3);

            } else if(K == 5){

                // /* Tuning with Unroll K = 5 */
                // loop p = 0, q = 0~4
                acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);
                acc += in_4d(b, c, S*h+0, S*w+3) * mask_4d(m, c, 0, 3);
                acc += in_4d(b, c, S*h+0, S*w+4) * mask_4d(m, c, 0, 4);

                // loop p = 1, q = 0~4
                acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);
                acc += in_4d(b, c, S*h+1, S*w+3) * mask_4d(m, c, 1, 3);
                acc += in_4d(b, c, S*h+1, S*w+4) * mask_4d(m, c, 1, 4);

                // loop p = 2, q = 0~4
                acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);
                acc += in_4d(b, c, S*h+2, S*w+3) * mask_4d(m, c, 2, 3);
                acc += in_4d(b, c, S*h+2, S*w+4) * mask_4d(m, c, 2, 4);

                // loop p = 3, q = 0~4
                acc += in_4d(b, c, S*h+3, S*w+0) * mask_4d(m, c, 3, 0);
                acc += in_4d(b, c, S*h+3, S*w+1) * mask_4d(m, c, 3, 1);
                acc += in_4d(b, c, S*h+3, S*w+2) * mask_4d(m, c, 3, 2);
                acc += in_4d(b, c, S*h+3, S*w+3) * mask_4d(m, c, 3, 3);
                acc += in_4d(b, c, S*h+3, S*w+4) * mask_4d(m, c, 3, 4);
        
                // loop p = 4, q = 0~4
                acc += in_4d(b, c, S*h+4, S*w+0) * mask_4d(m, c, 4, 0);
                acc += in_4d(b, c, S*h+4, S*w+1) * mask_4d(m, c, 4, 1);
                acc += in_4d(b, c, S*h+4, S*w+2) * mask_4d(m, c, 4, 2);
                acc += in_4d(b, c, S*h+4, S*w+3) * mask_4d(m, c, 4, 3);
                acc += in_4d(b, c, S*h+4, S*w+4) * mask_4d(m, c, 4, 4);

            } else if(K == 6){

                // /* Tuning with Unroll K = 6 */
                // loop p = 0, q = 0~5
                acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);
                acc += in_4d(b, c, S*h+0, S*w+3) * mask_4d(m, c, 0, 3);
                acc += in_4d(b, c, S*h+0, S*w+4) * mask_4d(m, c, 0, 4);
                acc += in_4d(b, c, S*h+0, S*w+5) * mask_4d(m, c, 0, 5);

                // loop p = 1, q = 0~5
                acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);
                acc += in_4d(b, c, S*h+1, S*w+3) * mask_4d(m, c, 1, 3);
                acc += in_4d(b, c, S*h+1, S*w+4) * mask_4d(m, c, 1, 4);
                acc += in_4d(b, c, S*h+1, S*w+5) * mask_4d(m, c, 1, 5);

                // loop p = 2, q = 0~5
                acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);
                acc += in_4d(b, c, S*h+2, S*w+3) * mask_4d(m, c, 2, 3);
                acc += in_4d(b, c, S*h+2, S*w+4) * mask_4d(m, c, 2, 4);
                acc += in_4d(b, c, S*h+2, S*w+5) * mask_4d(m, c, 2, 5);

                // loop p = 3, q = 0~5
                acc += in_4d(b, c, S*h+3, S*w+0) * mask_4d(m, c, 3, 0);
                acc += in_4d(b, c, S*h+3, S*w+1) * mask_4d(m, c, 3, 1);
                acc += in_4d(b, c, S*h+3, S*w+2) * mask_4d(m, c, 3, 2);
                acc += in_4d(b, c, S*h+3, S*w+3) * mask_4d(m, c, 3, 3);
                acc += in_4d(b, c, S*h+3, S*w+4) * mask_4d(m, c, 3, 4);
                acc += in_4d(b, c, S*h+3, S*w+5) * mask_4d(m, c, 3, 5);
        
                // loop p = 4, q = 0~5
                acc += in_4d(b, c, S*h+4, S*w+0) * mask_4d(m, c, 4, 0);
                acc += in_4d(b, c, S*h+4, S*w+1) * mask_4d(m, c, 4, 1);
                acc += in_4d(b, c, S*h+4, S*w+2) * mask_4d(m, c, 4, 2);
                acc += in_4d(b, c, S*h+4, S*w+3) * mask_4d(m, c, 4, 3);
                acc += in_4d(b, c, S*h+4, S*w+4) * mask_4d(m, c, 4, 4);
                acc += in_4d(b, c, S*h+4, S*w+5) * mask_4d(m, c, 4, 5);

                // loop p = 5, q = 0~5
                acc += in_4d(b, c, S*h+5, S*w+0) * mask_4d(m, c, 5, 0);
                acc += in_4d(b, c, S*h+5, S*w+1) * mask_4d(m, c, 5, 1);
                acc += in_4d(b, c, S*h+5, S*w+2) * mask_4d(m, c, 5, 2);
                acc += in_4d(b, c, S*h+5, S*w+3) * mask_4d(m, c, 5, 3);
                acc += in_4d(b, c, S*h+5, S*w+4) * mask_4d(m, c, 5, 4);
                acc += in_4d(b, c, S*h+5, S*w+5) * mask_4d(m, c, 5, 5);

            } else if(K == 7){

                // /* Tuning with Unroll K = 7 */
                // loop p = 0, q = 0~6
                acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);
                acc += in_4d(b, c, S*h+0, S*w+3) * mask_4d(m, c, 0, 3);
                acc += in_4d(b, c, S*h+0, S*w+4) * mask_4d(m, c, 0, 4);
                acc += in_4d(b, c, S*h+0, S*w+5) * mask_4d(m, c, 0, 5);
                acc += in_4d(b, c, S*h+0, S*w+6) * mask_4d(m, c, 0, 6);

                // loop p = 1, q = 0~6
                acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);
                acc += in_4d(b, c, S*h+1, S*w+3) * mask_4d(m, c, 1, 3);
                acc += in_4d(b, c, S*h+1, S*w+4) * mask_4d(m, c, 1, 4);
                acc += in_4d(b, c, S*h+1, S*w+5) * mask_4d(m, c, 1, 5);
                acc += in_4d(b, c, S*h+1, S*w+6) * mask_4d(m, c, 1, 6);

                // loop p = 2, q = 0~6
                acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);
                acc += in_4d(b, c, S*h+2, S*w+3) * mask_4d(m, c, 2, 3);
                acc += in_4d(b, c, S*h+2, S*w+4) * mask_4d(m, c, 2, 4);
                acc += in_4d(b, c, S*h+2, S*w+5) * mask_4d(m, c, 2, 5);
                acc += in_4d(b, c, S*h+2, S*w+6) * mask_4d(m, c, 2, 6);

                // loop p = 3, q = 0~6
                acc += in_4d(b, c, S*h+3, S*w+0) * mask_4d(m, c, 3, 0);
                acc += in_4d(b, c, S*h+3, S*w+1) * mask_4d(m, c, 3, 1);
                acc += in_4d(b, c, S*h+3, S*w+2) * mask_4d(m, c, 3, 2);
                acc += in_4d(b, c, S*h+3, S*w+3) * mask_4d(m, c, 3, 3);
                acc += in_4d(b, c, S*h+3, S*w+4) * mask_4d(m, c, 3, 4);
                acc += in_4d(b, c, S*h+3, S*w+5) * mask_4d(m, c, 3, 5);
                acc += in_4d(b, c, S*h+3, S*w+6) * mask_4d(m, c, 3, 6);
        
                // loop p = 4, q = 0~6
                acc += in_4d(b, c, S*h+4, S*w+0) * mask_4d(m, c, 4, 0);
                acc += in_4d(b, c, S*h+4, S*w+1) * mask_4d(m, c, 4, 1);
                acc += in_4d(b, c, S*h+4, S*w+2) * mask_4d(m, c, 4, 2);
                acc += in_4d(b, c, S*h+4, S*w+3) * mask_4d(m, c, 4, 3);
                acc += in_4d(b, c, S*h+4, S*w+4) * mask_4d(m, c, 4, 4);
                acc += in_4d(b, c, S*h+4, S*w+5) * mask_4d(m, c, 4, 5);
                acc += in_4d(b, c, S*h+4, S*w+6) * mask_4d(m, c, 4, 6);

                // loop p = 5, q = 0~6
                acc += in_4d(b, c, S*h+5, S*w+0) * mask_4d(m, c, 5, 0);
                acc += in_4d(b, c, S*h+5, S*w+1) * mask_4d(m, c, 5, 1);
                acc += in_4d(b, c, S*h+5, S*w+2) * mask_4d(m, c, 5, 2);
                acc += in_4d(b, c, S*h+5, S*w+3) * mask_4d(m, c, 5, 3);
                acc += in_4d(b, c, S*h+5, S*w+4) * mask_4d(m, c, 5, 4);
                acc += in_4d(b, c, S*h+5, S*w+5) * mask_4d(m, c, 5, 5);
                acc += in_4d(b, c, S*h+5, S*w+6) * mask_4d(m, c, 5, 6);

                // loop p = 6, q = 0~6
                acc += in_4d(b, c, S*h+6, S*w+0) * mask_4d(m, c, 6, 0);
                acc += in_4d(b, c, S*h+6, S*w+1) * mask_4d(m, c, 6, 1);
                acc += in_4d(b, c, S*h+6, S*w+2) * mask_4d(m, c, 6, 2);
                acc += in_4d(b, c, S*h+6, S*w+3) * mask_4d(m, c, 6, 3);
                acc += in_4d(b, c, S*h+6, S*w+4) * mask_4d(m, c, 6, 4);
                acc += in_4d(b, c, S*h+6, S*w+5) * mask_4d(m, c, 6, 5);
                acc += in_4d(b, c, S*h+6, S*w+6) * mask_4d(m, c, 6, 6);

            } else if(K == 8){                                                      // TESTED UNROLL IF K <= 8

                // /* Tuning with Unroll K = 8 */
                // loop p = 0, q = 0~7
                acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);
                acc += in_4d(b, c, S*h+0, S*w+3) * mask_4d(m, c, 0, 3);
                acc += in_4d(b, c, S*h+0, S*w+4) * mask_4d(m, c, 0, 4);
                acc += in_4d(b, c, S*h+0, S*w+5) * mask_4d(m, c, 0, 5);
                acc += in_4d(b, c, S*h+0, S*w+6) * mask_4d(m, c, 0, 6);
                acc += in_4d(b, c, S*h+0, S*w+7) * mask_4d(m, c, 0, 7);

                // loop p = 1, q = 0~7
                acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);
                acc += in_4d(b, c, S*h+1, S*w+3) * mask_4d(m, c, 1, 3);
                acc += in_4d(b, c, S*h+1, S*w+4) * mask_4d(m, c, 1, 4);
                acc += in_4d(b, c, S*h+1, S*w+5) * mask_4d(m, c, 1, 5);
                acc += in_4d(b, c, S*h+1, S*w+6) * mask_4d(m, c, 1, 6);
                acc += in_4d(b, c, S*h+1, S*w+7) * mask_4d(m, c, 1, 7);

                // loop p = 2, q = 0~7
                acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);
                acc += in_4d(b, c, S*h+2, S*w+3) * mask_4d(m, c, 2, 3);
                acc += in_4d(b, c, S*h+2, S*w+4) * mask_4d(m, c, 2, 4);
                acc += in_4d(b, c, S*h+2, S*w+5) * mask_4d(m, c, 2, 5);
                acc += in_4d(b, c, S*h+2, S*w+6) * mask_4d(m, c, 2, 6);
                acc += in_4d(b, c, S*h+2, S*w+7) * mask_4d(m, c, 2, 7);

                // loop p = 3, q = 0~7
                acc += in_4d(b, c, S*h+3, S*w+0) * mask_4d(m, c, 3, 0);
                acc += in_4d(b, c, S*h+3, S*w+1) * mask_4d(m, c, 3, 1);
                acc += in_4d(b, c, S*h+3, S*w+2) * mask_4d(m, c, 3, 2);
                acc += in_4d(b, c, S*h+3, S*w+3) * mask_4d(m, c, 3, 3);
                acc += in_4d(b, c, S*h+3, S*w+4) * mask_4d(m, c, 3, 4);
                acc += in_4d(b, c, S*h+3, S*w+5) * mask_4d(m, c, 3, 5);
                acc += in_4d(b, c, S*h+3, S*w+6) * mask_4d(m, c, 3, 6);
                acc += in_4d(b, c, S*h+3, S*w+7) * mask_4d(m, c, 3, 7);
        
                // loop p = 4, q = 0~7
                acc += in_4d(b, c, S*h+4, S*w+0) * mask_4d(m, c, 4, 0);
                acc += in_4d(b, c, S*h+4, S*w+1) * mask_4d(m, c, 4, 1);
                acc += in_4d(b, c, S*h+4, S*w+2) * mask_4d(m, c, 4, 2);
                acc += in_4d(b, c, S*h+4, S*w+3) * mask_4d(m, c, 4, 3);
                acc += in_4d(b, c, S*h+4, S*w+4) * mask_4d(m, c, 4, 4);
                acc += in_4d(b, c, S*h+4, S*w+5) * mask_4d(m, c, 4, 5);
                acc += in_4d(b, c, S*h+4, S*w+6) * mask_4d(m, c, 4, 6);
                acc += in_4d(b, c, S*h+4, S*w+7) * mask_4d(m, c, 4, 7);

                // loop p = 5, q = 0~7
                acc += in_4d(b, c, S*h+5, S*w+0) * mask_4d(m, c, 5, 0);
                acc += in_4d(b, c, S*h+5, S*w+1) * mask_4d(m, c, 5, 1);
                acc += in_4d(b, c, S*h+5, S*w+2) * mask_4d(m, c, 5, 2);
                acc += in_4d(b, c, S*h+5, S*w+3) * mask_4d(m, c, 5, 3);
                acc += in_4d(b, c, S*h+5, S*w+4) * mask_4d(m, c, 5, 4);
                acc += in_4d(b, c, S*h+5, S*w+5) * mask_4d(m, c, 5, 5);
                acc += in_4d(b, c, S*h+5, S*w+6) * mask_4d(m, c, 5, 6);
                acc += in_4d(b, c, S*h+5, S*w+7) * mask_4d(m, c, 5, 7);

                // loop p = 6, q = 0~7
                acc += in_4d(b, c, S*h+6, S*w+0) * mask_4d(m, c, 6, 0);
                acc += in_4d(b, c, S*h+6, S*w+1) * mask_4d(m, c, 6, 1);
                acc += in_4d(b, c, S*h+6, S*w+2) * mask_4d(m, c, 6, 2);
                acc += in_4d(b, c, S*h+6, S*w+3) * mask_4d(m, c, 6, 3);
                acc += in_4d(b, c, S*h+6, S*w+4) * mask_4d(m, c, 6, 4);
                acc += in_4d(b, c, S*h+6, S*w+5) * mask_4d(m, c, 6, 5);
                acc += in_4d(b, c, S*h+6, S*w+6) * mask_4d(m, c, 6, 6);
                acc += in_4d(b, c, S*h+6, S*w+7) * mask_4d(m, c, 6, 7);

                // loop p = 7, q = 0~7
                acc += in_4d(b, c, S*h+7, S*w+0) * mask_4d(m, c, 7, 0);
                acc += in_4d(b, c, S*h+7, S*w+1) * mask_4d(m, c, 7, 1);
                acc += in_4d(b, c, S*h+7, S*w+2) * mask_4d(m, c, 7, 2);
                acc += in_4d(b, c, S*h+7, S*w+3) * mask_4d(m, c, 7, 3);
                acc += in_4d(b, c, S*h+7, S*w+4) * mask_4d(m, c, 7, 4);
                acc += in_4d(b, c, S*h+7, S*w+5) * mask_4d(m, c, 7, 5);
                acc += in_4d(b, c, S*h+7, S*w+6) * mask_4d(m, c, 7, 6);
                acc += in_4d(b, c, S*h+7, S*w+7) * mask_4d(m, c, 7, 7);

            // } else if(K == 9){

            //     // /* Tuning with Unroll K = 9 */
            //     // loop p = 0, q = 0~8
            //     acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
            //     acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
            //     acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);
            //     acc += in_4d(b, c, S*h+0, S*w+3) * mask_4d(m, c, 0, 3);
            //     acc += in_4d(b, c, S*h+0, S*w+4) * mask_4d(m, c, 0, 4);
            //     acc += in_4d(b, c, S*h+0, S*w+5) * mask_4d(m, c, 0, 5);
            //     acc += in_4d(b, c, S*h+0, S*w+6) * mask_4d(m, c, 0, 6);
            //     acc += in_4d(b, c, S*h+0, S*w+7) * mask_4d(m, c, 0, 7);
            //     acc += in_4d(b, c, S*h+0, S*w+8) * mask_4d(m, c, 0, 8);

            //     // loop p = 1, q = 0~8
            //     acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
            //     acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
            //     acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);
            //     acc += in_4d(b, c, S*h+1, S*w+3) * mask_4d(m, c, 1, 3);
            //     acc += in_4d(b, c, S*h+1, S*w+4) * mask_4d(m, c, 1, 4);
            //     acc += in_4d(b, c, S*h+1, S*w+5) * mask_4d(m, c, 1, 5);
            //     acc += in_4d(b, c, S*h+1, S*w+6) * mask_4d(m, c, 1, 6);
            //     acc += in_4d(b, c, S*h+1, S*w+7) * mask_4d(m, c, 1, 7);
            //     acc += in_4d(b, c, S*h+1, S*w+8) * mask_4d(m, c, 1, 8);

            //     // loop p = 2, q = 0~8
            //     acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
            //     acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
            //     acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);
            //     acc += in_4d(b, c, S*h+2, S*w+3) * mask_4d(m, c, 2, 3);
            //     acc += in_4d(b, c, S*h+2, S*w+4) * mask_4d(m, c, 2, 4);
            //     acc += in_4d(b, c, S*h+2, S*w+5) * mask_4d(m, c, 2, 5);
            //     acc += in_4d(b, c, S*h+2, S*w+6) * mask_4d(m, c, 2, 6);
            //     acc += in_4d(b, c, S*h+2, S*w+7) * mask_4d(m, c, 2, 7);
            //     acc += in_4d(b, c, S*h+2, S*w+8) * mask_4d(m, c, 2, 8);

            //     // loop p = 3, q = 0~8
            //     acc += in_4d(b, c, S*h+3, S*w+0) * mask_4d(m, c, 3, 0);
            //     acc += in_4d(b, c, S*h+3, S*w+1) * mask_4d(m, c, 3, 1);
            //     acc += in_4d(b, c, S*h+3, S*w+2) * mask_4d(m, c, 3, 2);
            //     acc += in_4d(b, c, S*h+3, S*w+3) * mask_4d(m, c, 3, 3);
            //     acc += in_4d(b, c, S*h+3, S*w+4) * mask_4d(m, c, 3, 4);
            //     acc += in_4d(b, c, S*h+3, S*w+5) * mask_4d(m, c, 3, 5);
            //     acc += in_4d(b, c, S*h+3, S*w+6) * mask_4d(m, c, 3, 6);
            //     acc += in_4d(b, c, S*h+3, S*w+7) * mask_4d(m, c, 3, 7);
            //     acc += in_4d(b, c, S*h+3, S*w+8) * mask_4d(m, c, 3, 8);
        
            //     // loop p = 4, q = 0~8
            //     acc += in_4d(b, c, S*h+4, S*w+0) * mask_4d(m, c, 4, 0);
            //     acc += in_4d(b, c, S*h+4, S*w+1) * mask_4d(m, c, 4, 1);
            //     acc += in_4d(b, c, S*h+4, S*w+2) * mask_4d(m, c, 4, 2);
            //     acc += in_4d(b, c, S*h+4, S*w+3) * mask_4d(m, c, 4, 3);
            //     acc += in_4d(b, c, S*h+4, S*w+4) * mask_4d(m, c, 4, 4);
            //     acc += in_4d(b, c, S*h+4, S*w+5) * mask_4d(m, c, 4, 5);
            //     acc += in_4d(b, c, S*h+4, S*w+6) * mask_4d(m, c, 4, 6);
            //     acc += in_4d(b, c, S*h+4, S*w+7) * mask_4d(m, c, 4, 7);
            //     acc += in_4d(b, c, S*h+4, S*w+8) * mask_4d(m, c, 4, 8);

            //     // loop p = 5, q = 0~8
            //     acc += in_4d(b, c, S*h+5, S*w+0) * mask_4d(m, c, 5, 0);
            //     acc += in_4d(b, c, S*h+5, S*w+1) * mask_4d(m, c, 5, 1);
            //     acc += in_4d(b, c, S*h+5, S*w+2) * mask_4d(m, c, 5, 2);
            //     acc += in_4d(b, c, S*h+5, S*w+3) * mask_4d(m, c, 5, 3);
            //     acc += in_4d(b, c, S*h+5, S*w+4) * mask_4d(m, c, 5, 4);
            //     acc += in_4d(b, c, S*h+5, S*w+5) * mask_4d(m, c, 5, 5);
            //     acc += in_4d(b, c, S*h+5, S*w+6) * mask_4d(m, c, 5, 6);
            //     acc += in_4d(b, c, S*h+5, S*w+7) * mask_4d(m, c, 5, 7);
            //     acc += in_4d(b, c, S*h+5, S*w+8) * mask_4d(m, c, 5, 8);

            //     // loop p = 6, q = 0~8
            //     acc += in_4d(b, c, S*h+6, S*w+0) * mask_4d(m, c, 6, 0);
            //     acc += in_4d(b, c, S*h+6, S*w+1) * mask_4d(m, c, 6, 1);
            //     acc += in_4d(b, c, S*h+6, S*w+2) * mask_4d(m, c, 6, 2);
            //     acc += in_4d(b, c, S*h+6, S*w+3) * mask_4d(m, c, 6, 3);
            //     acc += in_4d(b, c, S*h+6, S*w+4) * mask_4d(m, c, 6, 4);
            //     acc += in_4d(b, c, S*h+6, S*w+5) * mask_4d(m, c, 6, 5);
            //     acc += in_4d(b, c, S*h+6, S*w+6) * mask_4d(m, c, 6, 6);
            //     acc += in_4d(b, c, S*h+6, S*w+7) * mask_4d(m, c, 6, 7);
            //     acc += in_4d(b, c, S*h+6, S*w+8) * mask_4d(m, c, 6, 8);

            //     // loop p = 7, q = 0~8
            //     acc += in_4d(b, c, S*h+7, S*w+0) * mask_4d(m, c, 7, 0);
            //     acc += in_4d(b, c, S*h+7, S*w+1) * mask_4d(m, c, 7, 1);
            //     acc += in_4d(b, c, S*h+7, S*w+2) * mask_4d(m, c, 7, 2);
            //     acc += in_4d(b, c, S*h+7, S*w+3) * mask_4d(m, c, 7, 3);
            //     acc += in_4d(b, c, S*h+7, S*w+4) * mask_4d(m, c, 7, 4);
            //     acc += in_4d(b, c, S*h+7, S*w+5) * mask_4d(m, c, 7, 5);
            //     acc += in_4d(b, c, S*h+7, S*w+6) * mask_4d(m, c, 7, 6);
            //     acc += in_4d(b, c, S*h+7, S*w+7) * mask_4d(m, c, 7, 7);
            //     acc += in_4d(b, c, S*h+7, S*w+8) * mask_4d(m, c, 7, 8);

            //     // loop p = 8, q = 0~8
            //     acc += in_4d(b, c, S*h+8, S*w+0) * mask_4d(m, c, 8, 0);
            //     acc += in_4d(b, c, S*h+8, S*w+1) * mask_4d(m, c, 8, 1);
            //     acc += in_4d(b, c, S*h+8, S*w+2) * mask_4d(m, c, 8, 2);
            //     acc += in_4d(b, c, S*h+8, S*w+3) * mask_4d(m, c, 8, 3);
            //     acc += in_4d(b, c, S*h+8, S*w+4) * mask_4d(m, c, 8, 4);
            //     acc += in_4d(b, c, S*h+8, S*w+5) * mask_4d(m, c, 8, 5);
            //     acc += in_4d(b, c, S*h+8, S*w+6) * mask_4d(m, c, 8, 6);
            //     acc += in_4d(b, c, S*h+8, S*w+7) * mask_4d(m, c, 8, 7);
            //     acc += in_4d(b, c, S*h+8, S*w+8) * mask_4d(m, c, 8, 8);

            // } else if(K == 10){                                                     // TESTED UNROLL IF K <= 10 OPTIMAL

            //     // /* Tuning with Unroll K = 10 */
            //     // loop p = 0, q = 0~9
            //     acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
            //     acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
            //     acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);
            //     acc += in_4d(b, c, S*h+0, S*w+3) * mask_4d(m, c, 0, 3);
            //     acc += in_4d(b, c, S*h+0, S*w+4) * mask_4d(m, c, 0, 4);
            //     acc += in_4d(b, c, S*h+0, S*w+5) * mask_4d(m, c, 0, 5);
            //     acc += in_4d(b, c, S*h+0, S*w+6) * mask_4d(m, c, 0, 6);
            //     acc += in_4d(b, c, S*h+0, S*w+7) * mask_4d(m, c, 0, 7);
            //     acc += in_4d(b, c, S*h+0, S*w+8) * mask_4d(m, c, 0, 8);
            //     acc += in_4d(b, c, S*h+0, S*w+9) * mask_4d(m, c, 0, 9);

            //     // loop p = 1, q = 0~9
            //     acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
            //     acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
            //     acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);
            //     acc += in_4d(b, c, S*h+1, S*w+3) * mask_4d(m, c, 1, 3);
            //     acc += in_4d(b, c, S*h+1, S*w+4) * mask_4d(m, c, 1, 4);
            //     acc += in_4d(b, c, S*h+1, S*w+5) * mask_4d(m, c, 1, 5);
            //     acc += in_4d(b, c, S*h+1, S*w+6) * mask_4d(m, c, 1, 6);
            //     acc += in_4d(b, c, S*h+1, S*w+7) * mask_4d(m, c, 1, 7);
            //     acc += in_4d(b, c, S*h+1, S*w+8) * mask_4d(m, c, 1, 8);
            //     acc += in_4d(b, c, S*h+1, S*w+9) * mask_4d(m, c, 1, 9);

            //     // loop p = 2, q = 0~9
            //     acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
            //     acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
            //     acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);
            //     acc += in_4d(b, c, S*h+2, S*w+3) * mask_4d(m, c, 2, 3);
            //     acc += in_4d(b, c, S*h+2, S*w+4) * mask_4d(m, c, 2, 4);
            //     acc += in_4d(b, c, S*h+2, S*w+5) * mask_4d(m, c, 2, 5);
            //     acc += in_4d(b, c, S*h+2, S*w+6) * mask_4d(m, c, 2, 6);
            //     acc += in_4d(b, c, S*h+2, S*w+7) * mask_4d(m, c, 2, 7);
            //     acc += in_4d(b, c, S*h+2, S*w+8) * mask_4d(m, c, 2, 8);
            //     acc += in_4d(b, c, S*h+2, S*w+9) * mask_4d(m, c, 2, 9);

            //     // loop p = 3, q = 0~9
            //     acc += in_4d(b, c, S*h+3, S*w+0) * mask_4d(m, c, 3, 0);
            //     acc += in_4d(b, c, S*h+3, S*w+1) * mask_4d(m, c, 3, 1);
            //     acc += in_4d(b, c, S*h+3, S*w+2) * mask_4d(m, c, 3, 2);
            //     acc += in_4d(b, c, S*h+3, S*w+3) * mask_4d(m, c, 3, 3);
            //     acc += in_4d(b, c, S*h+3, S*w+4) * mask_4d(m, c, 3, 4);
            //     acc += in_4d(b, c, S*h+3, S*w+5) * mask_4d(m, c, 3, 5);
            //     acc += in_4d(b, c, S*h+3, S*w+6) * mask_4d(m, c, 3, 6);
            //     acc += in_4d(b, c, S*h+3, S*w+7) * mask_4d(m, c, 3, 7);
            //     acc += in_4d(b, c, S*h+3, S*w+8) * mask_4d(m, c, 3, 8);
            //     acc += in_4d(b, c, S*h+3, S*w+9) * mask_4d(m, c, 3, 9);
        
            //     // loop p = 4, q = 0~9
            //     acc += in_4d(b, c, S*h+4, S*w+0) * mask_4d(m, c, 4, 0);
            //     acc += in_4d(b, c, S*h+4, S*w+1) * mask_4d(m, c, 4, 1);
            //     acc += in_4d(b, c, S*h+4, S*w+2) * mask_4d(m, c, 4, 2);
            //     acc += in_4d(b, c, S*h+4, S*w+3) * mask_4d(m, c, 4, 3);
            //     acc += in_4d(b, c, S*h+4, S*w+4) * mask_4d(m, c, 4, 4);
            //     acc += in_4d(b, c, S*h+4, S*w+5) * mask_4d(m, c, 4, 5);
            //     acc += in_4d(b, c, S*h+4, S*w+6) * mask_4d(m, c, 4, 6);
            //     acc += in_4d(b, c, S*h+4, S*w+7) * mask_4d(m, c, 4, 7);
            //     acc += in_4d(b, c, S*h+4, S*w+8) * mask_4d(m, c, 4, 8);
            //     acc += in_4d(b, c, S*h+4, S*w+9) * mask_4d(m, c, 4, 9);

            //     // loop p = 5, q = 0~9
            //     acc += in_4d(b, c, S*h+5, S*w+0) * mask_4d(m, c, 5, 0);
            //     acc += in_4d(b, c, S*h+5, S*w+1) * mask_4d(m, c, 5, 1);
            //     acc += in_4d(b, c, S*h+5, S*w+2) * mask_4d(m, c, 5, 2);
            //     acc += in_4d(b, c, S*h+5, S*w+3) * mask_4d(m, c, 5, 3);
            //     acc += in_4d(b, c, S*h+5, S*w+4) * mask_4d(m, c, 5, 4);
            //     acc += in_4d(b, c, S*h+5, S*w+5) * mask_4d(m, c, 5, 5);
            //     acc += in_4d(b, c, S*h+5, S*w+6) * mask_4d(m, c, 5, 6);
            //     acc += in_4d(b, c, S*h+5, S*w+7) * mask_4d(m, c, 5, 7);
            //     acc += in_4d(b, c, S*h+5, S*w+8) * mask_4d(m, c, 5, 8);
            //     acc += in_4d(b, c, S*h+5, S*w+9) * mask_4d(m, c, 5, 9);

            //     // loop p = 6, q = 0~9
            //     acc += in_4d(b, c, S*h+6, S*w+0) * mask_4d(m, c, 6, 0);
            //     acc += in_4d(b, c, S*h+6, S*w+1) * mask_4d(m, c, 6, 1);
            //     acc += in_4d(b, c, S*h+6, S*w+2) * mask_4d(m, c, 6, 2);
            //     acc += in_4d(b, c, S*h+6, S*w+3) * mask_4d(m, c, 6, 3);
            //     acc += in_4d(b, c, S*h+6, S*w+4) * mask_4d(m, c, 6, 4);
            //     acc += in_4d(b, c, S*h+6, S*w+5) * mask_4d(m, c, 6, 5);
            //     acc += in_4d(b, c, S*h+6, S*w+6) * mask_4d(m, c, 6, 6);
            //     acc += in_4d(b, c, S*h+6, S*w+7) * mask_4d(m, c, 6, 7);
            //     acc += in_4d(b, c, S*h+6, S*w+8) * mask_4d(m, c, 6, 8);
            //     acc += in_4d(b, c, S*h+6, S*w+9) * mask_4d(m, c, 6, 9);

            //     // loop p = 7, q = 0~9
            //     acc += in_4d(b, c, S*h+7, S*w+0) * mask_4d(m, c, 7, 0);
            //     acc += in_4d(b, c, S*h+7, S*w+1) * mask_4d(m, c, 7, 1);
            //     acc += in_4d(b, c, S*h+7, S*w+2) * mask_4d(m, c, 7, 2);
            //     acc += in_4d(b, c, S*h+7, S*w+3) * mask_4d(m, c, 7, 3);
            //     acc += in_4d(b, c, S*h+7, S*w+4) * mask_4d(m, c, 7, 4);
            //     acc += in_4d(b, c, S*h+7, S*w+5) * mask_4d(m, c, 7, 5);
            //     acc += in_4d(b, c, S*h+7, S*w+6) * mask_4d(m, c, 7, 6);
            //     acc += in_4d(b, c, S*h+7, S*w+7) * mask_4d(m, c, 7, 7);
            //     acc += in_4d(b, c, S*h+7, S*w+8) * mask_4d(m, c, 7, 8);
            //     acc += in_4d(b, c, S*h+7, S*w+9) * mask_4d(m, c, 7, 9);

            //     // loop p = 8, q = 0~9
            //     acc += in_4d(b, c, S*h+8, S*w+0) * mask_4d(m, c, 8, 0);
            //     acc += in_4d(b, c, S*h+8, S*w+1) * mask_4d(m, c, 8, 1);
            //     acc += in_4d(b, c, S*h+8, S*w+2) * mask_4d(m, c, 8, 2);
            //     acc += in_4d(b, c, S*h+8, S*w+3) * mask_4d(m, c, 8, 3);
            //     acc += in_4d(b, c, S*h+8, S*w+4) * mask_4d(m, c, 8, 4);
            //     acc += in_4d(b, c, S*h+8, S*w+5) * mask_4d(m, c, 8, 5);
            //     acc += in_4d(b, c, S*h+8, S*w+6) * mask_4d(m, c, 8, 6);
            //     acc += in_4d(b, c, S*h+8, S*w+7) * mask_4d(m, c, 8, 7);
            //     acc += in_4d(b, c, S*h+8, S*w+8) * mask_4d(m, c, 8, 8);
            //     acc += in_4d(b, c, S*h+8, S*w+9) * mask_4d(m, c, 8, 9);

            //     // loop p = 9, q = 0~9
            //     acc += in_4d(b, c, S*h+9, S*w+0) * mask_4d(m, c, 9, 0);
            //     acc += in_4d(b, c, S*h+9, S*w+1) * mask_4d(m, c, 9, 1);
            //     acc += in_4d(b, c, S*h+9, S*w+2) * mask_4d(m, c, 9, 2);
            //     acc += in_4d(b, c, S*h+9, S*w+3) * mask_4d(m, c, 9, 3);
            //     acc += in_4d(b, c, S*h+9, S*w+4) * mask_4d(m, c, 9, 4);
            //     acc += in_4d(b, c, S*h+9, S*w+5) * mask_4d(m, c, 9, 5);
            //     acc += in_4d(b, c, S*h+9, S*w+6) * mask_4d(m, c, 9, 6);
            //     acc += in_4d(b, c, S*h+9, S*w+7) * mask_4d(m, c, 9, 7);
            //     acc += in_4d(b, c, S*h+9, S*w+8) * mask_4d(m, c, 9, 8);
            //     acc += in_4d(b, c, S*h+9, S*w+9) * mask_4d(m, c, 9, 9);

            // } else if(K == 11){                                                         // TESTED UNROLL IF K <= 11

            //     // /* Tuning with Unroll K = 11 */
            //     // loop p = 0, q = 0~10
            //     acc += in_4d(b, c, S*h+0, S*w+0) * mask_4d(m, c, 0, 0);
            //     acc += in_4d(b, c, S*h+0, S*w+1) * mask_4d(m, c, 0, 1);
            //     acc += in_4d(b, c, S*h+0, S*w+2) * mask_4d(m, c, 0, 2);
            //     acc += in_4d(b, c, S*h+0, S*w+3) * mask_4d(m, c, 0, 3);
            //     acc += in_4d(b, c, S*h+0, S*w+4) * mask_4d(m, c, 0, 4);
            //     acc += in_4d(b, c, S*h+0, S*w+5) * mask_4d(m, c, 0, 5);
            //     acc += in_4d(b, c, S*h+0, S*w+6) * mask_4d(m, c, 0, 6);
            //     acc += in_4d(b, c, S*h+0, S*w+7) * mask_4d(m, c, 0, 7);
            //     acc += in_4d(b, c, S*h+0, S*w+8) * mask_4d(m, c, 0, 8);
            //     acc += in_4d(b, c, S*h+0, S*w+9) * mask_4d(m, c, 0, 9);
            //     acc += in_4d(b, c, S*h+0, S*w+10) * mask_4d(m, c, 0, 10);

            //     // loop p = 1, q = 0~10
            //     acc += in_4d(b, c, S*h+1, S*w+0) * mask_4d(m, c, 1, 0);
            //     acc += in_4d(b, c, S*h+1, S*w+1) * mask_4d(m, c, 1, 1);
            //     acc += in_4d(b, c, S*h+1, S*w+2) * mask_4d(m, c, 1, 2);
            //     acc += in_4d(b, c, S*h+1, S*w+3) * mask_4d(m, c, 1, 3);
            //     acc += in_4d(b, c, S*h+1, S*w+4) * mask_4d(m, c, 1, 4);
            //     acc += in_4d(b, c, S*h+1, S*w+5) * mask_4d(m, c, 1, 5);
            //     acc += in_4d(b, c, S*h+1, S*w+6) * mask_4d(m, c, 1, 6);
            //     acc += in_4d(b, c, S*h+1, S*w+7) * mask_4d(m, c, 1, 7);
            //     acc += in_4d(b, c, S*h+1, S*w+8) * mask_4d(m, c, 1, 8);
            //     acc += in_4d(b, c, S*h+1, S*w+9) * mask_4d(m, c, 1, 9);
            //     acc += in_4d(b, c, S*h+1, S*w+10) * mask_4d(m, c, 1, 10);

            //     // loop p = 2, q = 0~10
            //     acc += in_4d(b, c, S*h+2, S*w+0) * mask_4d(m, c, 2, 0);
            //     acc += in_4d(b, c, S*h+2, S*w+1) * mask_4d(m, c, 2, 1);
            //     acc += in_4d(b, c, S*h+2, S*w+2) * mask_4d(m, c, 2, 2);
            //     acc += in_4d(b, c, S*h+2, S*w+3) * mask_4d(m, c, 2, 3);
            //     acc += in_4d(b, c, S*h+2, S*w+4) * mask_4d(m, c, 2, 4);
            //     acc += in_4d(b, c, S*h+2, S*w+5) * mask_4d(m, c, 2, 5);
            //     acc += in_4d(b, c, S*h+2, S*w+6) * mask_4d(m, c, 2, 6);
            //     acc += in_4d(b, c, S*h+2, S*w+7) * mask_4d(m, c, 2, 7);
            //     acc += in_4d(b, c, S*h+2, S*w+8) * mask_4d(m, c, 2, 8);
            //     acc += in_4d(b, c, S*h+2, S*w+9) * mask_4d(m, c, 2, 9);
            //     acc += in_4d(b, c, S*h+2, S*w+10) * mask_4d(m, c, 2, 10);

            //     // loop p = 3, q = 0~10
            //     acc += in_4d(b, c, S*h+3, S*w+0) * mask_4d(m, c, 3, 0);
            //     acc += in_4d(b, c, S*h+3, S*w+1) * mask_4d(m, c, 3, 1);
            //     acc += in_4d(b, c, S*h+3, S*w+2) * mask_4d(m, c, 3, 2);
            //     acc += in_4d(b, c, S*h+3, S*w+3) * mask_4d(m, c, 3, 3);
            //     acc += in_4d(b, c, S*h+3, S*w+4) * mask_4d(m, c, 3, 4);
            //     acc += in_4d(b, c, S*h+3, S*w+5) * mask_4d(m, c, 3, 5);
            //     acc += in_4d(b, c, S*h+3, S*w+6) * mask_4d(m, c, 3, 6);
            //     acc += in_4d(b, c, S*h+3, S*w+7) * mask_4d(m, c, 3, 7);
            //     acc += in_4d(b, c, S*h+3, S*w+8) * mask_4d(m, c, 3, 8);
            //     acc += in_4d(b, c, S*h+3, S*w+9) * mask_4d(m, c, 3, 9);
            //     acc += in_4d(b, c, S*h+3, S*w+10) * mask_4d(m, c, 3, 10);

            //     // loop p = 4, q = 0~10
            //     acc += in_4d(b, c, S*h+4, S*w+0) * mask_4d(m, c, 4, 0);
            //     acc += in_4d(b, c, S*h+4, S*w+1) * mask_4d(m, c, 4, 1);
            //     acc += in_4d(b, c, S*h+4, S*w+2) * mask_4d(m, c, 4, 2);
            //     acc += in_4d(b, c, S*h+4, S*w+3) * mask_4d(m, c, 4, 3);
            //     acc += in_4d(b, c, S*h+4, S*w+4) * mask_4d(m, c, 4, 4);
            //     acc += in_4d(b, c, S*h+4, S*w+5) * mask_4d(m, c, 4, 5);
            //     acc += in_4d(b, c, S*h+4, S*w+6) * mask_4d(m, c, 4, 6);
            //     acc += in_4d(b, c, S*h+4, S*w+7) * mask_4d(m, c, 4, 7);
            //     acc += in_4d(b, c, S*h+4, S*w+8) * mask_4d(m, c, 4, 8);
            //     acc += in_4d(b, c, S*h+4, S*w+9) * mask_4d(m, c, 4, 9);
            //     acc += in_4d(b, c, S*h+4, S*w+10) * mask_4d(m, c, 4, 10);

            //     // loop p = 5, q = 0~10
            //     acc += in_4d(b, c, S*h+5, S*w+0) * mask_4d(m, c, 5, 0);
            //     acc += in_4d(b, c, S*h+5, S*w+1) * mask_4d(m, c, 5, 1);
            //     acc += in_4d(b, c, S*h+5, S*w+2) * mask_4d(m, c, 5, 2);
            //     acc += in_4d(b, c, S*h+5, S*w+3) * mask_4d(m, c, 5, 3);
            //     acc += in_4d(b, c, S*h+5, S*w+4) * mask_4d(m, c, 5, 4);
            //     acc += in_4d(b, c, S*h+5, S*w+5) * mask_4d(m, c, 5, 5);
            //     acc += in_4d(b, c, S*h+5, S*w+6) * mask_4d(m, c, 5, 6);
            //     acc += in_4d(b, c, S*h+5, S*w+7) * mask_4d(m, c, 5, 7);
            //     acc += in_4d(b, c, S*h+5, S*w+8) * mask_4d(m, c, 5, 8);
            //     acc += in_4d(b, c, S*h+5, S*w+9) * mask_4d(m, c, 5, 9);
            //     acc += in_4d(b, c, S*h+5, S*w+10) * mask_4d(m, c, 5, 10);

            //     // loop p = 6, q = 0~10
            //     acc += in_4d(b, c, S*h+6, S*w+0) * mask_4d(m, c, 6, 0);
            //     acc += in_4d(b, c, S*h+6, S*w+1) * mask_4d(m, c, 6, 1);
            //     acc += in_4d(b, c, S*h+6, S*w+2) * mask_4d(m, c, 6, 2);
            //     acc += in_4d(b, c, S*h+6, S*w+3) * mask_4d(m, c, 6, 3);
            //     acc += in_4d(b, c, S*h+6, S*w+4) * mask_4d(m, c, 6, 4);
            //     acc += in_4d(b, c, S*h+6, S*w+5) * mask_4d(m, c, 6, 5);
            //     acc += in_4d(b, c, S*h+6, S*w+6) * mask_4d(m, c, 6, 6);
            //     acc += in_4d(b, c, S*h+6, S*w+7) * mask_4d(m, c, 6, 7);
            //     acc += in_4d(b, c, S*h+6, S*w+8) * mask_4d(m, c, 6, 8);
            //     acc += in_4d(b, c, S*h+6, S*w+9) * mask_4d(m, c, 6, 9);
            //     acc += in_4d(b, c, S*h+6, S*w+10) * mask_4d(m, c, 6, 10);

            //     // loop p = 7, q = 0~10
            //     acc += in_4d(b, c, S*h+7, S*w+0) * mask_4d(m, c, 7, 0);
            //     acc += in_4d(b, c, S*h+7, S*w+1) * mask_4d(m, c, 7, 1);
            //     acc += in_4d(b, c, S*h+7, S*w+2) * mask_4d(m, c, 7, 2);
            //     acc += in_4d(b, c, S*h+7, S*w+3) * mask_4d(m, c, 7, 3);
            //     acc += in_4d(b, c, S*h+7, S*w+4) * mask_4d(m, c, 7, 4);
            //     acc += in_4d(b, c, S*h+7, S*w+5) * mask_4d(m, c, 7, 5);
            //     acc += in_4d(b, c, S*h+7, S*w+6) * mask_4d(m, c, 7, 6);
            //     acc += in_4d(b, c, S*h+7, S*w+7) * mask_4d(m, c, 7, 7);
            //     acc += in_4d(b, c, S*h+7, S*w+8) * mask_4d(m, c, 7, 8);
            //     acc += in_4d(b, c, S*h+7, S*w+9) * mask_4d(m, c, 7, 9);
            //     acc += in_4d(b, c, S*h+7, S*w+10) * mask_4d(m, c, 7, 10);

            //     // loop p = 8, q = 0~10
            //     acc += in_4d(b, c, S*h+8, S*w+0) * mask_4d(m, c, 8, 0);
            //     acc += in_4d(b, c, S*h+8, S*w+1) * mask_4d(m, c, 8, 1);
            //     acc += in_4d(b, c, S*h+8, S*w+2) * mask_4d(m, c, 8, 2);
            //     acc += in_4d(b, c, S*h+8, S*w+3) * mask_4d(m, c, 8, 3);
            //     acc += in_4d(b, c, S*h+8, S*w+4) * mask_4d(m, c, 8, 4);
            //     acc += in_4d(b, c, S*h+8, S*w+5) * mask_4d(m, c, 8, 5);
            //     acc += in_4d(b, c, S*h+8, S*w+6) * mask_4d(m, c, 8, 6);
            //     acc += in_4d(b, c, S*h+8, S*w+7) * mask_4d(m, c, 8, 7);
            //     acc += in_4d(b, c, S*h+8, S*w+8) * mask_4d(m, c, 8, 8);
            //     acc += in_4d(b, c, S*h+8, S*w+9) * mask_4d(m, c, 8, 9);
            //     acc += in_4d(b, c, S*h+8, S*w+10) * mask_4d(m, c, 8, 10);

            //     // loop p = 9, q = 0~10
            //     acc += in_4d(b, c, S*h+9, S*w+0) * mask_4d(m, c, 9, 0);
            //     acc += in_4d(b, c, S*h+9, S*w+1) * mask_4d(m, c, 9, 1);
            //     acc += in_4d(b, c, S*h+9, S*w+2) * mask_4d(m, c, 9, 2);
            //     acc += in_4d(b, c, S*h+9, S*w+3) * mask_4d(m, c, 9, 3);
            //     acc += in_4d(b, c, S*h+9, S*w+4) * mask_4d(m, c, 9, 4);
            //     acc += in_4d(b, c, S*h+9, S*w+5) * mask_4d(m, c, 9, 5);
            //     acc += in_4d(b, c, S*h+9, S*w+6) * mask_4d(m, c, 9, 6);
            //     acc += in_4d(b, c, S*h+9, S*w+7) * mask_4d(m, c, 9, 7);
            //     acc += in_4d(b, c, S*h+9, S*w+8) * mask_4d(m, c, 9, 8);
            //     acc += in_4d(b, c, S*h+9, S*w+9) * mask_4d(m, c, 9, 9);
            //     acc += in_4d(b, c, S*h+9, S*w+10) * mask_4d(m, c, 9, 10);

            //     // loop p = 10, q = 0~10
            //     acc += in_4d(b, c, S*h+10, S*w+0) * mask_4d(m, c, 10, 0);
            //     acc += in_4d(b, c, S*h+10, S*w+1) * mask_4d(m, c, 10, 1);
            //     acc += in_4d(b, c, S*h+10, S*w+2) * mask_4d(m, c, 10, 2);
            //     acc += in_4d(b, c, S*h+10, S*w+3) * mask_4d(m, c, 10, 3);
            //     acc += in_4d(b, c, S*h+10, S*w+4) * mask_4d(m, c, 10, 4);
            //     acc += in_4d(b, c, S*h+10, S*w+5) * mask_4d(m, c, 10, 5);
            //     acc += in_4d(b, c, S*h+10, S*w+6) * mask_4d(m, c, 10, 6);
            //     acc += in_4d(b, c, S*h+10, S*w+7) * mask_4d(m, c, 10, 7);
            //     acc += in_4d(b, c, S*h+10, S*w+8) * mask_4d(m, c, 10, 8);
            //     acc += in_4d(b, c, S*h+10, S*w+9) * mask_4d(m, c, 10, 9);
            //     acc += in_4d(b, c, S*h+10, S*w+10) * mask_4d(m, c, 10, 10);

            } else{
                for(int p = 0; p < K; p++){
                    for(int q = 0; q < K; q++){
                        acc += in_4d(b, c, S*h+p, S*w+q) * mask_4d(m, c, p, q);
                    }
                }
            }
        }
    out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    // find output width and height of each feature
    const int H_out = (H - K)/S + 1;      
    const int W_out = (W - K)/S + 1;      

    // find size of input, output, and mask
    int size_of_input = B*C*H*W*sizeof(float);
    int size_of_output = B*M*H_out*W_out*sizeof(float);
    int size_of_mask = M*C*K*K*sizeof(float);

    // allocate memory for input, output, and mask
    cudaMalloc((void**) device_input_ptr, size_of_input);
    cudaMalloc((void**) device_output_ptr, size_of_output);
    cudaMalloc((void**) device_mask_ptr, size_of_mask);

    // copy data to memory 
    cudaMemcpy(*device_input_ptr, host_input, size_of_input, cudaMemcpyHostToDevice);

    /* Constant Memory Optimization */
    // copy kernel mask values to constant memory
    cudaMemcpyToSymbol(weight_matrix, host_mask, size_of_mask);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;      
    const int W_out = (W - K)/S + 1;      
    float H_grid = ceil((float)H_out/TILE_WIDTH);      // # of vertical tiles per output map
    float W_grid = ceil((float)W_out/TILE_WIDTH);      // # of horizontal tiles per output map
    // int H_grid = ((H_out -1)/TILE_WIDTH) + 1;      // # of vertical tiles per output map
    // int W_grid = ((W_out -1)/TILE_WIDTH) + 1;      // # of horizontal tiles per output map
    dim3 dimGrid(M, H_grid*W_grid, B);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;      
    const int W_out = (W - K)/S + 1;      
    int size_of_output = B*M*H_out*W_out*sizeof(float);
    cudaMemcpy(host_output, device_output, size_of_output, cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

