#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void histogramKernel(uint* d_result, uint** d_data, int dataN,int BIN_COUNT){
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const int BC = 1024;
    __shared__ uint s_Hist[BC];
    
    for(int pos = threadIdx.x; pos<BC; pos+=blockDim.x){
        s_Hist[pos] = 0;
    }
    __syncthreads();
    for(int pos = globalTid; pos< dataN; pos+= numThreads){
        uint data4 = (d_data[pos/INPUT_HEIGHT][pos%INPUT_WIDTH]);
        uint data1 = (data4 >> 24) & 0xFFU;
        uint data2 = (data4 >> 16) & 0xFFU;
        uint data3 = (data4 >> 8) & 0xFFU;
        uint data5 = (data4 >> 0) & 0xFFU;
        atomicInc(s_Hist + data5,255);
        atomicInc(s_Hist + data3,255);
        atomicInc(s_Hist + data2,255);
        atomicInc(s_Hist + data1,255);
    }

    __syncthreads();
    for(int pos = threadIdx.x; pos<BC; pos+= blockDim.x){
      atomicAdd(d_result + pos, s_Hist[pos]);
    }
    for(int i = 0; i < 1024; i++) d_result[i] = d_data[i][i];
}

void opt_2dhisto(uint* d_result, uint** d_data, int dataN,int BIN_COUNT)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
       dim3 blockSize = (32);
       dim3 gridSize = (1);
       histogramKernel<<<blockSize,gridSize>>>(d_result, d_data, dataN,BIN_COUNT);
      //for(int i = 0; i < 256; i++) *(d_result + i) = i;
}
/* Include below the implementation of any other functions you need */
uint32_t** allocateInputOnDevice(uint32_t** hostInput,int height, int width){
    uint32_t** deviceInput = hostInput;
    int size = sizeof(uint32_t) * height * 4096;
    cudaMalloc((void**)deviceInput,size);
    cudaMemcpy(deviceInput,hostInput,size,cudaMemcpyHostToDevice);
    return deviceInput;
}

uint8_t* allocateHistogramOnDevice(uint8_t** hostHisto,int height,int width){
    uint8_t* deviceHisto = *hostHisto;
    cudaMalloc((void**)&deviceHisto,height*width*sizeof(uint8_t));
    cudaMemcpy(deviceHisto,hostHisto,height*width*sizeof(uint32_t),cudaMemcpyHostToDevice);
    return deviceHisto;
}


void cudaTeardown(uint8_t* deviceHisto, uint8_t*hostHisto,uint32_t** deviceInput){
    cudaDeviceSynchronize();
    cudaMemcpy(hostHisto,deviceHisto,1024*sizeof(uint8_t),cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceHisto);
}

