#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include <cmath>
// includes, kernels
#include <assert.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS) 

// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024

// Lab4: Host Helper Functions (allocate your own data structure...)

float* allocateDeviceArray(int size){
    float* deviceArray;
    cudaMalloc((void**)&deviceArray,size*sizeof(float));
    return deviceArray;
}

// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void localScanKernel(float *outArray, float *inArray, int numElements, float* rightmostArray) {
    __shared__ float inputSlice[BLOCK_SIZE + 1];
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = 0;

    if(blockIdx.x == 0) {
        inputSlice[0] = 0;
        offset = 1;
    }

    if(globalIndex > numElements - 1){ // Divergent but only for one block
        inputSlice[threadIdx.x + offset] = 0;
    } else {
        inputSlice[threadIdx.x + offset] = inArray[globalIndex - 1 + offset];
    }
    __syncthreads();

    int stride = 1;
    while (stride < BLOCK_SIZE) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < BLOCK_SIZE) {
            inputSlice[index] += inputSlice[index-stride];
        }
        stride = stride*2;
        __syncthreads();
    }

    stride = BLOCK_SIZE >> 1;
    while (stride > 0){
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index + stride < BLOCK_SIZE) {
            inputSlice[index+stride] += inputSlice[index];
        }
        stride = stride >> 1;
        __syncthreads();
    }
    
    if (globalIndex < numElements){
        outArray[threadIdx.x + blockIdx.x * blockDim.x] = inputSlice[threadIdx.x];
        if (threadIdx.x == 0 && blockIdx.x < gridDim.x - 1) rightmostArray[blockIdx.x] = inputSlice[BLOCK_SIZE - 1];
    }
}

__global__ void combineScansKernel(float *outArray, float *inArray, int numElements, float* rightmostArray){
    __shared__ float localEntries[BLOCK_SIZE];
    __shared__ float rightmostEntries[BLOCK_SIZE];
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (globalIndex > numElements - 1){
        localEntries[threadIdx.x] = 0;
    } else {
        localEntries[threadIdx.x] = inArray[threadIdx.x + blockDim.x * blockIdx.x];
    }

    // if (globalIndex < numElements) outArray[threadIdx.x + blockDim.x * blockIdx.x] = localEntries[threadIdx.x];

    int iLim = std::ceil((double)gridDim.x / (double)BLOCK_SIZE);
    for(int i = 0; i < iLim; i ++) {
        // Load in first BLOCK_SIZE elements of rightmostArray if they exist
        // (if ID < # of blocks - 1)
        int pos = threadIdx.x + i * blockDim.x;
        rightmostEntries[threadIdx.x] = (pos < gridDim.x - 1) ? rightmostArray[pos] : 0;
        // Add each element of the array while j < blockDim.x (only elements before) or
        // until we get to BLOCK_SIZE
        int val = blockIdx.x - i * BLOCK_SIZE;
        int jLim = (val > BLOCK_SIZE) ? BLOCK_SIZE : val;
        __syncthreads();
        for (int j = 0; j < jLim; j++) {
            localEntries[threadIdx.x] += rightmostEntries[j];
        }
        __syncthreads();
    }

    if (globalIndex < numElements) outArray[threadIdx.x + blockDim.x * blockIdx.x] = localEntries[threadIdx.x];
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements, float* rightmostArray, int gridSize)
{
    dim3 BlockDims(1024, 1);
    dim3 GridDims(gridSize, 1);
    localScanKernel<<<GridDims,BlockDims>>>(outArray,inArray,numElements,rightmostArray);
    //cudaDeviceSynchronize();
    combineScansKernel<<<GridDims,BlockDims>>>(outArray,outArray,numElements,rightmostArray);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
