#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void histogramKernel(uint* d_result, uint* d_data, int dataN,int BIN_COUNT){
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
   // const int BC = BIN_COUNT;
    __shared__ uint s_Hist[1024];
    
    for(int pos = threadIdx.x; pos<BIN_COUNT; pos+=blockDim.x){
        s_Hist[pos] = 0;
        __syncthreads();
    }
    for(int pos = globalTid; pos< dataN; pos+= numThreads){
        uint data4 = d_data[pos];
        atomicAdd(s_Hist + ((data4 >> 0) & 0xFFU),1);
        atomicAdd(s_Hist + ((data4 >> 8) & 0xFFU),1);
        atomicAdd(s_Hist + ((data4 >> 16) & 0xFFU),1);
        atomicAdd(s_Hist + ((data4 >> 24) & 0xFFU),1);
    }
    __syncthreads();
    for(int pos = threadIdx.x; pos<BIN_COUNT; pos+= blockDim.x){
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

uint32_t* initInput(uint32_t* hostInput){
    uint32_t* gpuInput;
    cudaMalloc((void**)&gpuInput,INPUT_WIDTH*INPUT_HEIGHT*sizeof(uint32_t));
    cudaMemcpy(gpuInput,hostInput,INPUT_WIDTH*INPUT_HEIGHT*sizeof(uint32_t),cudaMemcpyHostToDevice);
    return gpuInput;
}
uint8_t* initHisto(uint8_t* hostHisto){
    uint8_t* gpuHisto;
    cudaMalloc((void**)&gpuHisto,HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
    cudaMemcpy(gpuHisto,hostHisto,HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t),cudaMemcpyHostToDevice);
    return gpuHisto;
}
void cudaTeardown(uint8_t* deviceHisto, uint8_t*hostHisto,uint32_t* deviceInput){
    cudaDeviceSynchronize();
    cudaMemcpy(hostHisto,deviceHisto,HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t),cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceHisto);
}

