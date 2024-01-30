#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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

    const int H_out = (H - K)/S + 1;      // originally (H-K)/S + 1
    const int W_out = (W - K)/S + 1;      // originally (W-K)/S + 1
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working  

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = (W_out - 1)/TILE_WIDTH + 1;      // # of horizontal tiles per output map

    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;
    if(h < H_out && w < W_out){
        for(int c = 0; c < C; c++){
            for(int p = 0; p < K; p++){
                for(int q = 0; q < K; q++){
                    acc += in_4d(b, c, S*h+p, S*w+q) * mask_4d(m, c, p, q);
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

    // allocate device memory for input, output, and mask
    cudaMalloc((void**) device_input_ptr, size_of_input);
    cudaMalloc((void**) device_output_ptr, size_of_output);
    cudaMalloc((void**) device_mask_ptr, size_of_mask);






    /* Stream Optimization */
    // Per Campuswire TA/CA recommendation, set number of streams to be batch size B

    // pin host input and output memory; use size in bytes of input and output and default flag mode 
    cudaHostRegister((void*) host_input, size_of_input, cudaHostRegisterDefault);
    cudaHostRegister((void*) host_output, size_of_output, cudaHostRegisterDefault);

    // Set the kernel dimensions 
    float H_grid = ceil((float)H_out/TILE_WIDTH);      // # of vertical tiles per output map
    float W_grid = ceil((float)W_out/TILE_WIDTH);      // # of horizontal tiles per output map
    dim3 dimGrid(M, H_grid*W_grid, B/B);               // need to adjust grid dimension to account for segment size (divide by B (number of streams) for size of each segment)
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Declare and create streams (B streams per Campuswire recommendation)
    cudaStream_t stream[B];
    for(int i = 0; i < B; i++){
        cudaStreamCreate(&stream[i]);
    } 
    cudaDeviceSynchronize();

    // find segment offset for pointer
    int input_segment = (B*C*H*W)/B;
    int output_segment = (B*M*H_out*W_out)/B;

    // Transfer Data and Launch Kernel; Loop through each stream/segment
    for(int i = 0; i < B; i++){
        // copy input and mask data to device memory
        cudaMemcpyAsync(*device_mask_ptr, host_mask, size_of_mask, cudaMemcpyHostToDevice, stream[i]);
        // need to begin to transfer data from offset into the input array/matrix, which is given by the current stream segment (multiply by size of each segment)
        cudaMemcpyAsync(*device_input_ptr + i*input_segment, host_input + i*input_segment, input_segment*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        // launch kernel with current stream (offset into device output and input by current stream segment)
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(*device_output_ptr + i*output_segment, *device_input_ptr + i*input_segment, *device_mask_ptr, B, M, C, H, W, K, S);
        // copy output back to host memory 
        cudaMemcpyAsync((float*) host_output + i*output_segment, *device_output_ptr + i*output_segment, output_segment*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    // Destroy the streams
    for(int i = 0; i < B; i++){
        cudaStreamDestroy(stream[i]);
    }

    // unpin host input and output memory 
    cudaHostUnregister((void*) host_input);
    cudaHostUnregister((void*) host_output);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // // Set the kernel dimensions and call the kernel
    // const int H_out = (H - K)/S + 1;      
    // const int W_out = (W - K)/S + 1;      
    // float H_grid = ceil((float)H_out/TILE_WIDTH);      // # of vertical tiles per output map
    // float W_grid = ceil((float)W_out/TILE_WIDTH);      // # of horizontal tiles per output map
    // // int H_grid = ((H_out -1)/TILE_WIDTH) + 1;      // # of vertical tiles per output map
    // // int W_grid = ((W_out -1)/TILE_WIDTH) + 1;      // # of horizontal tiles per output map
    // dim3 dimGrid(M, H_grid*W_grid, B);
    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // // Copy the output back to host
    // const int H_out = (H - K)/S + 1;      
    // const int W_out = (W - K)/S + 1;      
    // int size_of_output = B*M*H_out*W_out*sizeof(float);
    // cudaMemcpy(host_output, device_output, size_of_output, cudaMemcpyDeviceToHost);
   
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