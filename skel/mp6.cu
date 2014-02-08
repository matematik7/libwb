#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define Mask_width  5
#define Mask_radius Mask_width/2

static const int BLOCK_SIZE = 16;
static const int IMAGE_TILE_SIZE = BLOCK_SIZE - 2*(Mask_radius);

//@@ INSERT CODE HERE

__device__ __host__ float clamp(float x, float start, float end) {
	return min(max(x, start), end);
}

__global__ void imageConvolution(int imageHeight, int imageWidth, int imageChannels,
	float * inputImageData, float * outputImageData, const float * __restrict__ maskData) {
	
	// j is column index
	const int j = threadIdx.x + blockIdx.x*IMAGE_TILE_SIZE;
	const int jLoad = j - Mask_radius;
	// i is row index
	const int i = threadIdx.y + blockIdx.y*IMAGE_TILE_SIZE;
	const int iLoad = i - Mask_radius;
	
	// load data for 3 channels, 0 if not in image
	__shared__ float inputTile[BLOCK_SIZE][BLOCK_SIZE][3];
	if (jLoad >= 0 && jLoad < imageWidth 
			&& iLoad >= 0 && iLoad < imageHeight) {
		for (int k = 0; k < imageChannels; k++) {
			inputTile[threadIdx.y][threadIdx.x][k] = 
				inputImageData[(iLoad * imageWidth + jLoad)*imageChannels + k];
		}
	} else {
		for (int k = 0; k < imageChannels; k++) {
			inputTile[threadIdx.y][threadIdx.x][k] = 0;
		}
	}
	
	__syncthreads();
	
	// computation for 3 channels
	if (j < imageWidth && i < imageHeight && threadIdx.x < IMAGE_TILE_SIZE && threadIdx.y < IMAGE_TILE_SIZE) {
		for (int k = 0; k < imageChannels; k++) {
			float accum = 0;
			for (int y = 0; y < Mask_width; y++) {
				for (int x = 0; x < Mask_width; x++) {
					int xOffset = threadIdx.x+x;
					int yOffset = threadIdx.y+y;
					float imagePixel = inputTile[yOffset][xOffset][k];
					float maskValue = maskData[y*Mask_width + x];
					accum += imagePixel * maskValue;
				}
			}
			outputImageData[(i * imageWidth + j) * imageChannels + k] = clamp(accum, 0, 1);
		}
	}
}

int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck(cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float)));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice));
    wbTime_stop(Copy, "Copying data to the GPU");

	// compute block and grid dimensions
	dim3 dimGrid((imageWidth-1)/IMAGE_TILE_SIZE + 1, (imageHeight-1)/IMAGE_TILE_SIZE + 1, 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    wbTime_start(Compute, "Doing the computation on the GPU");
    
    imageConvolution<<<dimGrid, dimBlock>>>(imageHeight, imageWidth, imageChannels, 
		deviceInputImageData, deviceOutputImageData, deviceMaskData);
    
    wbCheck(cudaPeekAtLastError());
    wbCheck(cudaDeviceSynchronize());
    
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
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
