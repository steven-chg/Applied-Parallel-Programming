// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

// Originally 512 changed to 1024 to handle maximum input size of 2048*2048 
// (to be able to handle 2nd scan on auxiliary array with a single block)
#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add(float *block_sum, float *input_output, int len) {

  // since each block processes 2 * its block size elements
  unsigned int start = BLOCK_SIZE*2*blockIdx.x;

  // don't need to modify the first block 
  if(blockIdx.x > 0){
    // add the sum of the previous block to all elements in current blocks
    if((start + threadIdx.x) < len){
      input_output[start + threadIdx.x] += block_sum[blockIdx.x - 1];
    }
    if((start + BLOCK_SIZE + threadIdx.x) < len){
      input_output[start + BLOCK_SIZE + threadIdx.x] += block_sum[blockIdx.x - 1];
    }
  }
}

// __global__ void scan_one(float *input, float *output, float *auxiliary_array, int len) {
  
//   // load input into shared memory T
//   __shared__ float T[BLOCK_SIZE*2];

//   // Since each block processes 2 * its block size elements
//   unsigned int start = BLOCK_SIZE*2*blockIdx.x; 
  
//   // Copy over current segment of input into shared memory
//   // Account for possible out of bound input access 
//   if((start + threadIdx.x) < len){
//     T[threadIdx.x] = input[start + threadIdx.x];
//   } else{
//     T[threadIdx.x] = 0.0f;
//   }

//   if((BLOCK_SIZE + start + threadIdx.x) < len){
//     T[BLOCK_SIZE + threadIdx.x] = input[start + BLOCK_SIZE + threadIdx.x];
//   } else{
//     T[BLOCK_SIZE + threadIdx.x] = 0.0f;
//   }

//   // scan step
//   int stride = 1;
//   while(stride < BLOCK_SIZE*2){
//     __syncthreads();
//     int index = (threadIdx.x + 1)*stride*2 - 1;
//     if(index < BLOCK_SIZE*2 && (index - stride) >= 0){
//       T[index] += T[index - stride];
//     }
//     stride = stride*2;
//   }

//   // post scan step
//   int stride_post = BLOCK_SIZE/2;
//   while(stride_post > 0){
//     __syncthreads();
//     int index = (threadIdx.x + 1)*stride_post*2 - 1;
//     if((index + stride_post) < BLOCK_SIZE*2){
//       T[index + stride_post] += T[index];
//     }
//     stride_post = stride_post/2;
//   }

//   // fill current block scan result into correct block in output
//   __syncthreads(); // just in case, to make sure that the shared memory has correct output
//   if((start + threadIdx.x) < len){
//     output[start + threadIdx.x] = T[threadIdx.x];
//   }

//   if((BLOCK_SIZE + start + threadIdx.x) < len){
//     output[start + BLOCK_SIZE + threadIdx.x] = T[BLOCK_SIZE + threadIdx.x];
//   }

//   // set the auxiliary array for the current block total sum (from last element in shared memory)
//   auxiliary_array[blockIdx.x] = T[BLOCK_SIZE*2 - 1];
// }

__global__ void scan(float *input, float *output, float *auxiliary_array, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  // load input into shared memory T
  __shared__ float T[BLOCK_SIZE*2];

  // Since each block processes 2 * its block size elements
  unsigned int start = BLOCK_SIZE*2*blockIdx.x; 
  
  // Copy over current segment of input into shared memory
  // Account for possible out of bound input access 
  if(start + threadIdx.x < len){
    T[threadIdx.x] = input[start + threadIdx.x];
  } else{
    T[threadIdx.x] = 0.0f;
  }

  if((start + BLOCK_SIZE + threadIdx.x) < len){
    T[BLOCK_SIZE + threadIdx.x] = input[start + BLOCK_SIZE + threadIdx.x];
  } else{
    T[BLOCK_SIZE + threadIdx.x] = 0.0f;
  }

  // scan step
  int stride = 1;
  while(stride < BLOCK_SIZE*2){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if(index < BLOCK_SIZE*2 && (index - stride) >= 0){
      T[index] += T[index - stride];
    }
    stride = stride*2;
  }

  // post scan step
  int stride_post = BLOCK_SIZE/2;
  while(stride_post > 0){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride_post*2 - 1;
    if((index + stride_post) < BLOCK_SIZE*2){
      T[index + stride_post] += T[index];
    }
    stride_post = stride_post/2;
  }

  // fill current block scan result into correct block in output
  __syncthreads(); // just in case, to make sure that the shared memory has correct output
  if(start + threadIdx.x < len){
    output[start + threadIdx.x] = T[threadIdx.x];
  }

  if((start + BLOCK_SIZE + threadIdx.x) < len){
    output[start + BLOCK_SIZE + threadIdx.x] = T[BLOCK_SIZE + threadIdx.x];
  }

  __syncthreads(); // since we also pass in the auxiliary_array the second time as well
  // set the auxiliary array for the current block total sum (from last element in shared memory) if there is an auxiliary array input 
  auxiliary_array[blockIdx.x] = T[BLOCK_SIZE*2 - 1];


}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // find the necessary size of the auxiliary array
  int sizeofauxiliary_array;
  sizeofauxiliary_array = numElements/(BLOCK_SIZE*2); //BLOCK_SIZE*2 because each block can handle BLOCK_SIZE*2 elements
  if(numElements % (BLOCK_SIZE*2) != 0){
    sizeofauxiliary_array++;
  }

  // allocate global/device memory for auxiliary array 
  float *auxiliary_array;
  cudaMalloc((void**) &auxiliary_array, sizeofauxiliary_array*sizeof(float));

  // allocate global/device memory for scan block sum (CONSIDER USER CONSTANT MEMORY?)
  float *scan_block_sum;
  cudaMalloc((void**) &scan_block_sum, sizeofauxiliary_array*sizeof(float));

  //@@ Initialize the grid and block dimensions here

  // Scan Kernel 1
  dim3 dimGrid(sizeofauxiliary_array, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  // Scan Kernel 2
  dim3 dimGrid_two(1, 1, 1);
  dim3 dimBlock_two(sizeofauxiliary_array, 1, 1);

  // Add Kernel
  dim3 dimGrid_add(sizeofauxiliary_array, 1, 1);
  dim3 dimBlock_add(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  // Scan Kernel Launch 1
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, auxiliary_array, numElements);
  cudaDeviceSynchronize();

  /* After first scan, auxiliary array is filled and output is filled (block by block so inaccurate) */

  // Scan Kernel Launch 2
  scan<<<dimGrid_two, dimBlock_two>>>(auxiliary_array, scan_block_sum, auxiliary_array, sizeofauxiliary_array); // just pass the auxiliary_array back in (which will change it but it doesn't matter )
  cudaDeviceSynchronize();

  /* After second scan, scan_block_sum ready to be added to deviceOutput */
  add<<<dimGrid_add, dimBlock_add>>>(scan_block_sum, deviceOutput, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(auxiliary_array);    // free auxiliary array
  cudaFree(scan_block_sum);     // free scan block sum
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
