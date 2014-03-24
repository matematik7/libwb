#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

//@@ INSERT CODE HERE
static const int BLOCK_SIZE = 16;
static const int LIN_BLOCK_SIZE = 256;
static const int LOOP_SIZE = 8;

__global__ void floatToUchar(int height, int width, int channels, float * input, unsigned char * uchar) {
	int x = blockIdx.x * LIN_BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * LOOP_SIZE;
	
	if (x < width*channels) {
		for (; y < (blockIdx.y + 1)*LOOP_SIZE; y++) {
			if (y < height)
				uchar[y*width*channels + x] = (unsigned char) (255*input[y*width*channels + x]);
		}
	}
}

__global__ void ucharToFloat(int height, int width, int channels, float * output, unsigned char * uchar) {
	int x = blockIdx.x * LIN_BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * LOOP_SIZE;
	
	if (x < width*channels) {
		for (; y < (blockIdx.y + 1)*LOOP_SIZE; y++) {
			if (y < height)
				output[y*width*channels + x] = (float) (uchar[y*width*channels + x] / 255.0f);
		}
	}
}

__global__ void toGrayscale(int height, int width, int channels, unsigned char uchar, unsigned char gray) {
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int idx = y*width + x;
	unsigned char r = uchar[3*idx];
	unsigned char g = uchar[3*idx + 1];
	unsigned char b = uchar[3*idx + 2];
	gray[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
}

int main(int argc, char* argv[]) {

    wbArg_t arg = wbArg_read(argc, argv); /* parse the input arguments */

    char *inputImageFile = wbArg_getInputFile(arg, 0);

    wbImage_t inputImage = wbImport(inputImageFile);

 
    int imageWidth = wbImage_getWidth(inputImage);
    int imageHeight = wbImage_getHeight(inputImage);
    int imageChannels = wbImage_getChannels(inputImage);

    wbImage_t outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    float * hostInputImageData = wbImage_getData(inputImage);
    float * hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    float * deviceInputImageData;
    float * deviceOutputImageData;
    unsigned char * deviceUcharImage;
    unsigned char * deviceUcharGrayImage;
    wbCheck(cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceUcharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char)));
    wbCheck(cudaMalloc((void **) &deviceUcharGrayImage, imageWidth * imageHeight * sizeof(unsigned char)));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice));
	wbTime_stop(Copy, "Copying data to the GPU");

	// compute block and grid dimensions
	dim3 linearGrid((imageWidth*imageChannels-1)/LIN_BLOCK_SIZE + 1, (imageHeight-1)/LOOP_SIZE + 1, 1);
	dim3 linearBlock(LIN_BLOCK_SIZE, 1, 1);
	dim3 imageGrid((imageWidth - 1)/BLOCK_SIZE + 1, (imageHeight - 1)/BLOCK_SIZE + 1, 1);
	dim3 imageBlock(BLOCK_SIZE, BLOCK_SIZE);
		
    wbTime_start(Compute, "Doing the computation on the GPU");
    
    wbTime_start(Compute, "Float to unsigned char");
    floatToUchar<<<linearGrid, linearBlock>>>(imageHeight, imageWidth, imageChannels, 
		deviceInputImageData, deviceUcharImage);
    
    wbCheck(cudaPeekAtLastError());
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Float to unsigned char");

    wbTime_start(Compute, "To grayscale");
    toGrayscale<<<imageGrid, imageBlock>>>(imageHeight, imageWidth, imageChannels, 
		deviceUcharImage, deviceUcharGrayImage);
    
    wbCheck(cudaPeekAtLastError());
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "To grayscale");

    wbTime_start(Compute, "Unsigned char to float");
    ucharToFloat<<<linearGrid, linearBlock>>>(imageHeight, imageWidth, imageChannels, 
		deviceOutputImageData, deviceUcharImage);
    
    wbCheck(cudaPeekAtLastError());
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Unsigned char to float");
    
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck(cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceUcharImage);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
