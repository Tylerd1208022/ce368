#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 64

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void localScanKernel(float *outArray, float *inArray,int numElements){
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float inputSlice[BLOCK_SIZE];
    inputSlice[threadIdx.x] = inArray[threadIdx.x];
    //inputSlice[32 + threadIdx.x] = inArray[globalIndex*2] + inArray[globalIndex*2 + 1];
    __syncthreads();

    for(int i = 1; i < BLOCK_SIZE; i *= 2){
        int index = i * threadIdx.x * 2;
        if (index < blockDim.x){
            inputSlice[BLOCK_SIZE - index - 1] += inputSlice[BLOCK_SIZE - index - i - 1];
        }
        __syncthreads();
    }
    
    int stride = BLOCK_SIZE >> 1;
    while (stride > 0){
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index + stride < BLOCK_SIZE) {
            inputSlice[index+stride] += inputSlice[index];
        }
        stride = stride >> 1;
        __syncthreads();
    }
    outArray[threadIdx.x] = inputSlice[threadIdx.x];
    //outArray[BLOCK_SIZE + threadIdx.x] = inputSlice[BLOCK_SIZE  + threadIdx.x];

}
__global__ void combineScansKernel(float *outArray, float *inArray,int numElements){

}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
    dim3 BlockDims(64);
    dim3 GridDims(1);
    localScanKernel<<<GridDims,BlockDims>>>(outArray,inArray,numElements);

}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
