#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include <cmath>
// includes, kernels
#include <assert.h>

// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define CONFLICT_FREE_OFFSET(n) ((n) + (n)/NUM_BANKS) 

// Lab4: Host Helper Functions (allocate your own data structure...)

float* allocateDeviceArray(int size){
    float* deviceArray;
    cudaMalloc((void**)&deviceArray,size*sizeof(float));
    return deviceArray;
}

// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void localScanKernel(float *outArray, float *inArray, int numElements, float* rightmostArray) {
    __shared__ float inputSlice[BLOCK_SIZE + 1 + BLOCK_SIZE/NUM_BANKS];
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = 0;

    if (blockIdx.x == 0) {
        inputSlice[0] = 0;
        offset = 1;
    }

    if (globalIndex > numElements - 1){ // Divergent but only for one block
        inputSlice[CONFLICT_FREE_OFFSET(threadIdx.x + offset)] = 0;
    } else {
        inputSlice[CONFLICT_FREE_OFFSET(threadIdx.x + offset)] = inArray[globalIndex - 1 + offset];
    }
    __syncthreads();

    int stride = 1;
    while (stride < BLOCK_SIZE) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < BLOCK_SIZE) {
            inputSlice[CONFLICT_FREE_OFFSET(index)] += inputSlice[CONFLICT_FREE_OFFSET(index-stride)];
        }
        stride = stride*2;
        __syncthreads();
    }

    stride = BLOCK_SIZE >> 1;
    while (stride > 0){
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index + stride < BLOCK_SIZE) {
            inputSlice[CONFLICT_FREE_OFFSET(index+stride)] += inputSlice[CONFLICT_FREE_OFFSET(index)];
        }
        stride = stride >> 1;
        __syncthreads();
    }
    
    if (globalIndex < numElements){
        outArray[threadIdx.x + blockIdx.x * blockDim.x] = inputSlice[CONFLICT_FREE_OFFSET(threadIdx.x)];
        if (threadIdx.x == 0 && blockIdx.x < gridDim.x - 1) rightmostArray[blockIdx.x] = inputSlice[CONFLICT_FREE_OFFSET(BLOCK_SIZE - 1)];
    }
}

__global__ void combineScansKernel(float *outArray, float *inArray, int numElements, float* rightmostArray){
    __shared__ float localEntries[BLOCK_SIZE];
    __shared__ float rightmostEntry;
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;

    rightmostEntry = rightmostArray[blockIdx.x];

    if (globalIndex > numElements - 1){
        localEntries[threadIdx.x] = 0;
    } else {
        localEntries[threadIdx.x] = inArray[threadIdx.x + blockDim.x * blockIdx.x];
    }

    localEntries[threadIdx.x] += rightmostEntry;
        
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

    size_t rightmost_len = std::ceil((double)numElements/(double)BLOCK_SIZE);
    size_t mem_size = sizeof(float) * rightmost_len;
    float* h_rightmost_sum = (float*) malloc(mem_size);
    float* h_rightmost = (float*) malloc(mem_size);
    cudaMemcpy(h_rightmost, rightmostArray, mem_size, cudaMemcpyDeviceToHost);
    for( unsigned int i = 1; i < rightmost_len; ++i) 
    {
        h_rightmost_sum[i] = h_rightmost_sum[i-1] + h_rightmost[i-1];
    }
    cudaMemcpy(rightmostArray, h_rightmost_sum, mem_size, cudaMemcpyHostToDevice);
    
    combineScansKernel<<<GridDims,BlockDims>>>(outArray,outArray,numElements,rightmostArray);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
