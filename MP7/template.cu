// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256                  // self defined
#define HALF_HISTOGRAM_LENGTH 128       // self defined

//@@ insert code here

  /* Kernel function that converts between unsigned char* and float* */
  __global__ void float_tofrom_char(int width, int height, int channels, float* floatImage, unsigned char* unsignedCharImage, int flag){
    
    // find index into image 
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // if flag == 1, perform float* to unsigned char* 
    if(flag == 1){
      // check bounds
      if(idx < width*height*channels){
        unsignedCharImage[idx] = (unsigned char) (255 * floatImage[idx]);
      }
      return;
    }

    // if flag == 0, perform unsigned char* to float*
    if(flag == 0){
      // check bounds
      if(idx < width*height*channels){
        floatImage[idx] = (float) (unsignedCharImage[idx]/255.0);
      }
      return;
    }
  }

  /* Kernel function that converts an image from RGB to grayscale */
  __global__ void RGB_to_grayscale(int width, int height, unsigned char* unsignedCharImage, unsigned char* grayImage){

    // find index into image
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // convert from RGB to grayscale
    if(idx < width*height){
        unsigned char r = unsignedCharImage[3*idx];
        unsigned char g = unsignedCharImage[3*idx + 1];
        unsigned char b = unsignedCharImage[3*idx + 2];
        grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }

  }

  /* Kernel function that computes the histogram of the image */
  __global__ void histogram_compute(unsigned char* buffer, long size, unsigned int* histo){
    __shared__ unsigned int private_histo[HISTOGRAM_LENGTH]; // make sure number of threads > 256

    if(threadIdx.x < HISTOGRAM_LENGTH){
      private_histo[threadIdx.x] = 0;
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // stride = total number of threads
    int stride = blockDim.x * gridDim.x;

    while(i < size){
      atomicAdd(&(private_histo[buffer[i]]), 1);
      i += stride;
    }

    __syncthreads();

    if(threadIdx.x < 256){
      atomicAdd(&(histo[threadIdx.x]), private_histo[threadIdx.x]);
    }
  }

  /* Kernel function that computes the CDF of the histogram (scan operation) */
  __global__ void histogram_cdf(unsigned int* histo, float* cdf, int len, int width, int height){

    /* ASSUME WE INITITATE BLOCK DIMENSION TO BE HALF OF HISTOGRAM LENGTH */

    // load input into shared memory T
    __shared__ float T[HISTOGRAM_LENGTH];
  
    // Copy over current segment of input into shared memory
    // Account for possible out of bound input access 
    if(threadIdx.x < len){
      T[threadIdx.x] = histo[threadIdx.x];
    } else{
      T[threadIdx.x] = 0.0f;
    }

    if((HALF_HISTOGRAM_LENGTH + threadIdx.x) < len){
      T[HALF_HISTOGRAM_LENGTH + threadIdx.x] = histo[HALF_HISTOGRAM_LENGTH + threadIdx.x];
    } else{
      T[HALF_HISTOGRAM_LENGTH + threadIdx.x] = 0.0f;
    }

    // scan step
    int stride = 1;
    while(stride < HISTOGRAM_LENGTH){
      __syncthreads();
      int index = (threadIdx.x + 1)*stride*2 - 1;
      if(index < HISTOGRAM_LENGTH && (index - stride) >= 0){
        T[index] += T[index - stride];
      }
      stride = stride*2;
    }

    // post scan step
    int stride_post = HALF_HISTOGRAM_LENGTH/2;
    while(stride_post > 0){
      __syncthreads();
      int index = (threadIdx.x + 1)*stride_post*2 - 1;
      if((index + stride_post) < HISTOGRAM_LENGTH){
        T[index + stride_post] += T[index];
      }
      stride_post = stride_post/2;
    }

    // fill current block scan result into correct block in output
    __syncthreads(); // just in case, to make sure that the shared memory has correct output
    if(threadIdx.x < len){
      cdf[threadIdx.x] = (float) ((T[threadIdx.x] * 1.0)/(1.0 * width * height));
    }

    if((HALF_HISTOGRAM_LENGTH + threadIdx.x) < len){
      cdf[HALF_HISTOGRAM_LENGTH + threadIdx.x] = (float) ((T[HALF_HISTOGRAM_LENGTH + threadIdx.x] * 1.0)/(1.0 * width * height));
    }
  }

  /* Kernel function that applies the correct_color() function to apply histogram equalization function */
  __global__ void histogram_equalization_applied(int width, int height, int channels, unsigned char* unsignedCharImage, float* cdf){
    
    // find index into image
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // check bounds
    if(idx < width*height*channels){
      unsigned char cdfmin = cdf[0];
      unsigned char val = unsignedCharImage[idx];
      unsignedCharImage[idx] = (unsigned char) min(max(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0.0), 255.0);
    }
  }

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  /* Declare Pointers */
  float* deviceFloatInputImageData;                 // initial input image data (float*)
  unsigned char* deviceUnsignedImageData;           // image data (converted from float* to unsigned char*)
  unsigned char* deviceGrayscaleImageData;          // image data (converted from RGB to grayscale)
  unsigned int* deviceHistogram;                    // histogram
  float* deviceCdf;                                 // cdf
  float* deviceFloatOutputImageData;                // output (float*)



  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here

  int imageSize = imageWidth * imageHeight * imageChannels;
  int imageFloatSize = imageWidth * imageHeight * imageChannels * sizeof(float);
  int imageCharSize = imageWidth * imageHeight * imageChannels * sizeof(unsigned char);

  /* Allocate Device Memory */
  cudaMalloc((void**) &deviceFloatInputImageData, imageFloatSize);
  cudaMalloc((void**) &deviceUnsignedImageData, imageCharSize);
  cudaMalloc((void**) &deviceGrayscaleImageData, imageCharSize/imageChannels);
  cudaMalloc((void**) &deviceHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMalloc((void**) &deviceCdf, HISTOGRAM_LENGTH*sizeof(float));
  cudaMalloc((void**) &deviceFloatOutputImageData, imageFloatSize);

  /* Set Up: Copy Host to Device Memory */
  cudaMemcpy(deviceFloatInputImageData, hostInputImageData, imageFloatSize, cudaMemcpyHostToDevice);

  /* Kernel Function 1: Convert image from float* to unsigned char* */
  dim3 dimGridFtoC(ceil((1.0*imageSize)/BLOCK_SIZE), 1, 1);
  dim3 dimBlockFtoC(BLOCK_SIZE, 1, 1);
  float_tofrom_char<<<dimGridFtoC, dimBlockFtoC>>>(imageWidth, imageHeight, imageChannels, deviceFloatInputImageData, deviceUnsignedImageData, 1); 
  cudaDeviceSynchronize();

  /* Kernel Function 2: Convert image from RGB to grayscale */
  dim3 dimGridRGBtoG(ceil((1.0*(imageSize/imageChannels))/BLOCK_SIZE), 1, 1);
  dim3 dimBlockRGBtoG(BLOCK_SIZE, 1, 1);
  RGB_to_grayscale<<<dimGridRGBtoG, dimBlockRGBtoG>>>(imageWidth, imageHeight, deviceUnsignedImageData, deviceGrayscaleImageData);
  cudaDeviceSynchronize();

  /* Kernel Function 3: Compute histogram of grayscale image */
  dim3 dimGridHistogram(ceil((1.0*(imageSize/imageChannels))/HISTOGRAM_LENGTH), 1, 1);
  dim3 dimBlockHistogram(HISTOGRAM_LENGTH, 1, 1);
  histogram_compute<<<dimGridHistogram, dimBlockHistogram>>>(deviceGrayscaleImageData, imageWidth * imageHeight, deviceHistogram);
  cudaDeviceSynchronize();

  /* Kernel Function 4: Compute the cdf of the histogram */
  dim3 dimGridCdf(1, 1, 1);
  dim3 dimBlockCdf(HALF_HISTOGRAM_LENGTH, 1, 1);
  histogram_cdf<<<dimGridCdf, dimBlockCdf>>>(deviceHistogram, deviceCdf, HISTOGRAM_LENGTH, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  /* Kernel Function 5: Apply histogram equalization */
  dim3 dimGridHistoEqual(ceil((1.0*imageSize)/BLOCK_SIZE), 1, 1);
  dim3 dimBlockHistoEqual(BLOCK_SIZE, 1, 1);
  histogram_equalization_applied<<<dimGridHistoEqual, dimBlockHistoEqual>>>(imageWidth, imageHeight, imageChannels, deviceUnsignedImageData, deviceCdf);
  cudaDeviceSynchronize();

  /* Kernel Function 6: Convert image from unsigned char* to float* */
  dim3 dimGridCtoF(ceil((1.0*imageSize)/BLOCK_SIZE), 1, 1);
  dim3 dimBlockCtoF(BLOCK_SIZE, 1, 1);
  float_tofrom_char<<<dimGridCtoF, dimBlockCtoF>>>(imageWidth, imageHeight, imageChannels, deviceFloatOutputImageData, deviceUnsignedImageData, 0);
  cudaDeviceSynchronize();

  /* Copy Device to Host Output */
  cudaMemcpy(hostOutputImageData, deviceFloatOutputImageData, imageFloatSize, cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here

  /* Free Device Memory */
  cudaFree(deviceFloatInputImageData);
  cudaFree(deviceUnsignedImageData);
  cudaFree(deviceGrayscaleImageData);
  cudaFree(deviceHistogram);
  cudaFree(deviceCdf);
  cudaFree(deviceFloatOutputImageData);

  return 0;
}
