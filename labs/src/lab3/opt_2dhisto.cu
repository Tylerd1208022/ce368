#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void histogramKernel(uint* d_result, uint* d_data, int dataN,int BIN_COUNT){
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const int BC = 1024;
    __shared__ uint s_Hist[BC];
    
    for(int pos = threadIdx.x; pos<BC; pos+=blockDim.x){
        s_Hist[pos] = 0;
        
    }
    
    __syncthreads();
    for(int pos = globalTid; pos< dataN; pos+= numThreads){
        uint data4 = d_data[pos];
        /*atomicAdd(s_Hist + ((data4 >> 0) & 0xFFU),1);
        atomicAdd(s_Hist + ((data4 >> 8) & 0xFFU),1);
        atomicAdd(s_Hist + ((data4 >> 16) & 0xFFU),1);
        atomicAdd(s_Hist + ((data4 >> 24) & 0xFFU),1);*/
        atomicInc(s_Hist + ((data4 >> 0) & 0xFFU),255);
        atomicInc(s_Hist + ((data4 >> 8) & 0xFFU),255);
        atomicInc(s_Hist + ((data4 >> 16) & 0xFFU),255);
        atomicInc(s_Hist + ((data4 >> 24) & 0xFFU),255);
    }

    __syncthreads();
    for(int pos = threadIdx.x; pos<BC; pos+= blockDim.x){
        
        atomicAdd(d_result + pos, s_Hist[pos]);
    }
}

void opt_2dhisto(uint* d_result, uint* d_data, int dataN,int BIN_COUNT)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
       const int blockSize = 32;
       const int gridSize = 1;
       histogramKernel<<<blockSize,gridSize>>>(d_result, d_data, dataN,BIN_COUNT);

}
/* Include below the implementation of any other functions you need */
uint32_t* allocateInputOnDevice(uint32_t* hostInput){
    uint32_t* deviceInput;
    cudaMalloc((void**)&deviceInput,INPUT_WIDTH*INPUT_HEIGHT*sizeof(uint32_t));
    return deviceInput;
}

void copyInputToDevice(uint32_t* deviceInput, uint32_t* hostInput){
    cudaMemcpy(deviceInput,hostInput,INPUT_WIDTH*INPUT_HEIGHT*sizeof(uint32_t),cudaMemcpyHostToDevice);
}

uint8_t* allocateHistogramOnDevice(uint8_t* hostHisto){
    uint8_t* deviceHisto;
    cudaMalloc((void**)&deviceHisto,HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
    return deviceHisto;
}
void copyHistoToDevice(uint8_t* hostHisto, uint8_t* deviceHisto){
    cudaMemcpy(deviceHisto,hostHisto,HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t),cudaMemcpyHostToDevice);
}

void cudaTeardown(uint8_t* deviceHisto, uint8_t*hostHisto,uint32_t* deviceInput){
    cudaDeviceSynchronize();
    cudaMemcpy(hostHisto,deviceHisto,HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t),cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceHisto);
}

