#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include <vector>
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

    localScanKernel<<<GridDims,BlockDims>>>(outArray, inArray, numElements, rightmostArray);

    int n = std::ceil((double)numElements / (double)BLOCK_SIZE);
    std::vector<float*> rightArrays = {rightmostArray};
    std::vector<int> num_elements = {n};
    while (n > 2 * BLOCK_SIZE) {
        float* rArray = allocateDeviceArray(std::ceil((double)n/(double)BLOCK_SIZE));

        localScanKernel<<<GridDims,BlockDims>>>(rightArrays.back(), rightArrays.back(), n, rArray);
    
        n = std::ceil((double)n / (double)BLOCK_SIZE);
        rightArrays.push_back(rArray);
        num_elements.push_back(n);
    }

    size_t n_size = sizeof(float) * n;
    float* h_rightmost_sum = (float*) malloc(n_size);
    float* h_rightmost = (float*) malloc(n_size);
    cudaMemcpy(h_rightmost, rightArrays.back(), n_size, cudaMemcpyDeviceToHost);
    h_rightmost_sum[0] = 0;
    for(unsigned int i = 1; i < n; ++i) 
    {
        h_rightmost_sum[i] = h_rightmost_sum[i-1] + h_rightmost[i-1];
    }
    cudaMemcpy(rightArrays.back(), h_rightmost_sum, n_size, cudaMemcpyHostToDevice);
    
    for (int i = rightArrays.size() - 2; i >= 0; i--) {
        combineScansKernel<<<GridDims,BlockDims>>>(rightArrays[i], rightArrays[i], num_elements[i], rightArrays[i + 1]);
    }
    combineScansKernel<<<GridDims,BlockDims>>>(outArray, outArray, numElements, rightArrays[0]);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
