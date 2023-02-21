#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__global__ void zeroKernel(uint* d_result){ 
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const int BIN_COUNT = 1024;

    // initialize d_result to 0
    for(int pos = globalTid; pos < BIN_COUNT; pos += numThreads){
        d_result[pos] = 0;
    }
}

__global__ void histogramKernel(uint* d_result, uint* d_data, int dataN){ 
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const int BIN_COUNT = 1024;
    __shared__ uint s_Hist[BIN_COUNT + 32];

    // initialize hist to 0
    for(int pos = threadIdx.x; pos < BIN_COUNT + 32; pos += blockDim.x){
        s_Hist[pos] = 0;
    }

    __syncthreads();

    // compute hist
    for(int pos = globalTid; pos < INPUT_HEIGHT * INPUT_WIDTH; pos += numThreads){
        int x = pos % INPUT_WIDTH;
        int y = pos / INPUT_WIDTH;
        uint data = d_data[x + (y * 4096)];
        int shX = data%33;
        int shY = data/33;
        atomicAdd(s_Hist + 33*shY + shX, 1);
    }

    __syncthreads();

    // merge bins
    for(int pos = threadIdx.x * 4; pos + 4 <= BIN_COUNT + 32; pos += blockDim.x * 4){
        bool written = false;
        while (!written) {
            uint merged_bin = 0;
            int shX = pos%33;
            int shY = pos/33;
            int shPosn = shX + shY * 33;
            
            uint currDVal = d_result[pos/4];
            uint currD1 = currDVal & 0xFF;
            uint currD2 = (currDVal >> 8) & 0xFF;
            uint currD3 = (currDVal >> 16) & 0xFF;
            uint currD4 = (currDVal >> 24) & 0xFF;

            merged_bin += (currD1 + s_Hist[shPosn] > 255 ? 255 : currD1 + s_Hist[shPosn]);
            merged_bin += (currD2 + s_Hist[shPosn + 1] > 255 ? 255 : currD2 + s_Hist[shPosn + 1]) << 8;
            merged_bin += (currD3 + s_Hist[shPosn + 2] > 255 ? 255 : currD3 + s_Hist[shPosn + 2]) << 16;
            merged_bin += (currD4 + s_Hist[shPosn + 3] > 255 ? 255 : currD4 + s_Hist[shPosn + 3]) << 24;

            uint res = atomicCAS(d_result + pos/4, currDVal, merged_bin);
            if (res == currDVal)
                written = true;
        }
    }
}

void opt_2dhisto(uint* d_result, uint* d_data, int dataN, uint* kernel_bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    dim3 blockSize(32,1,1);
    dim3 gridSize(32,1);

    zeroKernel<<<gridSize,blockSize>>>(d_result);
    histogramKernel<<<gridSize,blockSize>>>(d_result, d_data, dataN);

    cudaDeviceSynchronize();
    cudaMemcpy(kernel_bins,d_result,1024*sizeof(uint8_t),cudaMemcpyDeviceToHost);

    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
}

/* Include below the implementation of any other functions you need */
uint32_t* allocateInputOnDevice(uint32_t** hostInput, int height, int width){
    // 1D Alloc
    uint32_t* pointer = *hostInput;
    cudaMalloc((void**)&pointer,4*4096*4096);
    cudaMemcpy(pointer,*hostInput,sizeof(uint32_t)*4096*4096,cudaMemcpyHostToDevice);
    return pointer;
}

uint8_t* allocateHistogramOnDevice(uint8_t** hostHisto, int height, int width){
    uint8_t* deviceHisto = *hostHisto;
    cudaMalloc((void**)&deviceHisto,height*width*sizeof(uint8_t));
    cudaMemcpy(deviceHisto,hostHisto,height*width*sizeof(uint32_t),cudaMemcpyHostToDevice);
    return deviceHisto;
}


void cudaTeardown(uint8_t* deviceHisto, uint8_t*hostHisto, uint32_t* deviceInput){
    cudaFree(deviceHisto);
    cudaFree(deviceInput);
}

