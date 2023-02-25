#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 32

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void localScanKernel(float *outArray, float *inArray,int numElements){
    // int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float inputSlice[BLOCK_SIZE + 1];
    inputSlice[0] = 0;
    inputSlice[threadIdx.x + 1] = inArray[threadIdx.x];
    __syncthreads();

    int stride = 1;
    while (stride < BLOCK_SIZE) {
        int idx = (threadIdx.x + 1) * stride * 2;
        if (idx < BLOCK_SIZE){
            inputSlice[idx] += inputSlice[idx - stride];
        }
        stride = stride * 2;
        __syncthreads();
    }

    stride = BLOCK_SIZE >> 1;
    while (stride > 0){
        int idx = (threadIdx.x + 1) * stride * 2;
        if(idx + stride < BLOCK_SIZE) {
            inputSlice[idx + stride] += inputSlice[idx];
        }
        stride = stride >> 1;
        __syncthreads();
    }

    outArray[threadIdx.x] = inputSlice[threadIdx.x];
}

__global__ void combineScansKernel(float *outArray, float *inArray,int numElements){

}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
    dim3 BlockDims(32, 1);
    dim3 GridDims(1, 1);
    localScanKernel<<<GridDims,BlockDims>>>(outArray,inArray,numElements);

}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
