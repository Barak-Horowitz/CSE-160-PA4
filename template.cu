#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
#define tileWidth 16 

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
	// store thread indices in registers for quick access
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bxd = blockDim.x; int bx = blockIdx.x;
	int byd = blockDim.y; int by = blockIdx.y; 
        // calculate current row and column
	int currRow = byd * by + ty;
	int currColumn = bxd * bx + tx;	
	// stores width in register for quick access.
	int width = numAColumns;
	// create 2D arrays representing tile for matrix A, and B.
	__shared__ float matrixATile[tileWidth][tileWidth];
	__shared__ float matrixBTile[tileWidth][tileWidth];

	float finalVal = 0;
	// loop over every phase of multiplication - EACH ITERATION COMBINES TWO SUBMATRICES!
	for(int numTileMultiplies = 0; numTileMultiplies < 1 + (width - 1) / tileWidth; numTileMultiplies ++) {
	       if(currRow < numCRows && numTileMultiplies * tileWidth + tx < width) {
	  	       	// indices for A: A(currRow, numTileMultiplies* tileWidth + tx)
	       		// loads proper ROW element into A subtile(Transposed for more efficient retrieval)
			matrixATile[tx][ty] = A[(currRow * width) + (numTileMultiplies*tileWidth) + tx];
		} else {
			matrixATile[tx][ty] = 0;
		}
		if(currColumn < numCColumns && numTileMultiplies * tileWidth + ty < width) {
			// loads proper COLUMN element into B subtile
			//indices for B:(numTileMultiplies * tileWidth + ty, currColumn)
			matrixBTile[ty][tx] = B[(numTileMultiplies * tileWidth + ty) * width + currColumn];
		} else {
			matrixBTile[ty][tx] = 0;
		}
		//ensures matrixATile and matrixBTile are fully loaded properly before multiplications
		__syncthreads();
            	// multiply tiles together
		for(int k = 0; k < tileWidth; k++) {
			finalVal += matrixATile[k][ty] * matrixBTile[k][tx];
		}
		// ensures finalVal stores correct value before progressing in the loop.
		__syncthreads();
	}
	// store final computation in matrix C
	if(currRow < numCRows && currColumn < numCColumns) {
       		C[currRow * numCColumns + currColumn] = finalVal;
	}


}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");
  hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice);
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 threadsPerBlock(tileWidth,tileWidth);
  dim3 blocksPerGrid(numCRows/tileWidth + 1, numCColumns/tileWidth + 1);
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC,
							   numARows, numAColumns,
							   numBRows, numBColumns,
							   numCRows, numCColumns);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);
 
  gpuTKTime_stop(GPU, "Freeing GPU Memory");
  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
