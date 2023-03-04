#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
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
__global__ void localScanKernel(float *outArray, float *inArray,int numElements,float* rightmostArray){
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float inputSlice[BLOCK_SIZE + 1];
    int offset = 0;
    if(blockIdx.x == 0) {
        inputSlice[0] = 0;
        offset = 1;
    }
    if(globalIndex > numElements - 1){//Divergent but only for one block
        inputSlice[threadIdx.x + offset] = 0;
    } else {
        inputSlice[threadIdx.x + offset] = inArray[globalIndex - 1 + offset];
    }
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
    if (globalIndex < numElements){
        outArray[threadIdx.x + blockIdx.x * blockDim.x] = inputSlice[threadIdx.x];
        if (threadIdx.x == 0 && blockIdx.x < gridDim.x - 1) rightmostArray[blockIdx.x] = inputSlice[BLOCK_SIZE - 1];
    }
    //outArray[BLOCK_SIZE + threadIdx.x] = inputSlice[BLOCK_SIZE  + threadIdx.x];

}

__global__ void combineScansKernel(float *outArray, float *inArray,int numElements,float* rightmostArray){
    
    __shared__ float localEntries[1024];
    __shared__ float rightmostEntries[1024];
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalIndex > numElements - 1){
        localEntries[threadIdx.x] = 0;
    }else {
        localEntries[threadIdx.x] = inArray[threadIdx.x + blockDim.x * blockIdx.x];
    }
    if (globalIndex < numElements) outArray[threadIdx.x  + blockDim.x * blockIdx.x] = localEntries[threadIdx.x];
    int val = gridDim.x % 1024;
    int iLim = (val == 0) ? (gridDim.x/ 1024) : (gridDim.x / 1024) + 1;
    for(int i = 0; i < iLim; i ++){//
        //Load in first 1024 elements of rightmostArray if they exist
        //(if ID < # of blocks - 1)
        int pos = threadIdx.x + i * blockDim.x;
        rightmostEntries[threadIdx.x] = (pos < gridDim.x - 1) ? rightmostArray[threadIdx.x + i * blockDim.x] : 0;
        //Add each element of the array while j < blockDim.x (only elements before) or
        //until we get to 1024
        int val = blockIdx.x - i * BLOCK_SIZE;
        int jLim = (val > 1024) ? 1024 : val;
        __syncthreads();
        for (int j = 0; j < jLim; j++){
            const int val = rightmostEntries[j];
            localEntries[threadIdx.x] +=  val;//1014;//rightmostEntries[0];
        }
        __syncthreads();
       // outArray[2047] = rightmostEntries[0];
    }
    if (globalIndex < numElements) outArray[threadIdx.x  + blockDim.x * blockIdx.x] = localEntries[threadIdx.x];
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements,float* rightmostArray, int gridSize)
{
    dim3 BlockDims(1024, 1);
    dim3 GridDims(gridSize, 1);
    localScanKernel<<<GridDims,BlockDims>>>(outArray,inArray,numElements,rightmostArray);
    //cudaDeviceSynchronize();
    combineScansKernel<<<GridDims,BlockDims>>>(outArray,outArray,numElements,rightmostArray);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
