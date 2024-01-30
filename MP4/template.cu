#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
const int MASK_WIDTH = 3;
const int TILE_WIDTH = 5;   

//@@ Define constant memory for device kernel here
__constant__ float mask[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

  //Utilizing strategy 2 extended to support 3 dimensions

  //Allocate shared memory for input matrix of size TILE_WIDTH + MASK_WIDTH - 1
  __shared__ float inputTile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

  //Declare and initilize block and thread indices in x, y, z
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  //Find row, column, aisles of input and output matrix
  int row_o = by*TILE_WIDTH + ty;
  int col_o = bx*TILE_WIDTH + tx;
  int aisle_o = bz*TILE_WIDTH + tz;

  int row_i = row_o - (MASK_WIDTH / 2);
  int col_i = col_o - (MASK_WIDTH / 2);
  int aisle_i = aisle_o - (MASK_WIDTH / 2);

  //Initialize running sum to 0
  float Pvalue = 0.0;

  //Fill in the input tile
  if((row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size) && (aisle_i >= 0) && (aisle_i < z_size)){
    inputTile[ty][tx][tz] = input[aisle_i*x_size*y_size + row_i*x_size + col_i];
  }
  else{
    inputTile[ty][tx][tz] = 0.0f;
  }

  __syncthreads();

  //Calculate output running sum 
  if(ty < TILE_WIDTH && tx < TILE_WIDTH && tz < TILE_WIDTH){
    //Triple nested for loop to go through 3 dimensions
    for(int i = 0; i < MASK_WIDTH; i++){
      for(int j = 0; j < MASK_WIDTH; j++){
        for(int k = 0; k < MASK_WIDTH; k++){
          Pvalue += mask[i][j][k] * inputTile[i+ty][j+tx][k+tz];
        }
      }
    }
    //After calculating the sum for each entry in the matrix, insert it into the output matrix
    if(row_o < y_size && col_o < x_size && aisle_o < z_size){
      output[aisle_o*x_size*y_size + row_o*x_size + col_o] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;  //kernelLength = 3^3 = 27 (total number of elements)
  float *hostInput;
  float *hostKernel;              //holds the mask matrix
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  //Allocate GPU memory for input and output 3d matrices
  cudaMalloc((void **) &deviceInput, (inputLength-3)*sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength-3)*sizeof(float));

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  //Copying hostKernel to device memory/GPU
  cudaMemcpyToSymbol(mask, hostKernel, kernelLength*sizeof(float));
  //Copy hostInput to GPU (pointer to fourth element of hostInput)
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size/(1.0*TILE_WIDTH)), ceil(y_size/(1.0*TILE_WIDTH)), ceil(z_size/(1.0*TILE_WIDTH)));
  dim3 DimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  //Copy to hostOutput from device (pointer to fourth element of hostOutput)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
