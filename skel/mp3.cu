#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                     \
        }                                                  \
    } while(0)

// compute matrix size in ram from dimensions
#define MATRIX_SIZE(a,b) ((a) * (b) * sizeof(float))

#define BLOCK_SIZE 16
#define TILE_SIZE BLOCK_SIZE

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    // j is column index
	const int j = threadIdx.x + blockDim.x * blockIdx.x;
	// i is row index
	const int i = threadIdx.y + blockDim.y * blockIdx.y;
	const int numTiles = (numAColumns-1)/TILE_SIZE + 1;
	__shared__ float tileA[TILE_SIZE][TILE_SIZE];
	__shared__ float tileB[TILE_SIZE][TILE_SIZE];
	
	float sum = 0;
	for (int k = 0; k < numTiles; k++) {
		// load from A
		const int aColumn = k*TILE_SIZE + threadIdx.x;
		if ((i < numARows) && (aColumn < numAColumns)) {
			tileA[threadIdx.y][threadIdx.x] = A[i * numAColumns + aColumn];
		} else {
			tileA[threadIdx.y][threadIdx.x] = 0;
		}
		// load from B
		const int bRow = threadIdx.y + k*TILE_SIZE;
		if ((bRow < numBRows) && (j < numBColumns)) {
			tileB[threadIdx.y][threadIdx.x] = B[bRow * numBColumns + j];
		} else {
			tileB[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
		// calculate
		for (int n = 0; n < TILE_SIZE; n++) {
			sum += tileA[threadIdx.y][n] * tileB[n][threadIdx.x];
		}
		__syncthreads();
	}
	
	// if we calculated valid element, save it
	if ((i < numCRows) && (j < numCColumns)) {
		C[i * numCColumns + j] = sum;
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    
    // calculate all sizes
    const size_t sizeA = MATRIX_SIZE(numARows, numAColumns);
    const size_t sizeB = MATRIX_SIZE(numBRows, numBColumns);
    const size_t sizeC = MATRIX_SIZE(numCRows, numCColumns);

    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(sizeC);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc((void **) &deviceA, sizeA));
	wbCheck(cudaMalloc((void **) &deviceB, sizeB));
	wbCheck(cudaMalloc((void **) &deviceC, sizeC));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
	dim3 dimGrid((numCColumns-1)/BLOCK_SIZE + 1, (numCRows-1)/BLOCK_SIZE + 1, 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
     
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, 
		numARows, numAColumns,
		numBRows, numBColumns,
		numCRows, numCColumns
	);
	
	wbCheck(cudaPeekAtLastError());
	
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceA));
    wbCheck(cudaFree(deviceB));
    wbCheck(cudaFree(deviceC));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

