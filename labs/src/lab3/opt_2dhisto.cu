#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void histogramKernel(uint* d_result, uint* d_data, int dataN, int BIN_COUNT){
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const int BC = 1024;
    __shared__ uint s_Hist[BC];
    __shared__ int sum;
    for(int pos = threadIdx.x; pos<BC; pos+=blockDim.x){
        s_Hist[pos] = 0;
    }
    __syncthreads();
    for(int pos = threadIdx.x; pos < INPUT_HEIGHT * INPUT_WIDTH; pos += 32){
        int x = pos%INPUT_WIDTH;
        int y = pos/INPUT_WIDTH;
        uint data = d_data[x + (y*4096)];
        if (s_Hist[data] < 255) atomicAdd(s_Hist + data,1);
    }
    __syncthreads();
    //atomicAdd(&sum,1);
    for(int i = 0; i < 1024; i++) s_Hist[i] = i;
    for(int pos = threadIdx.x * 4; pos + 4 <= 1024; pos += blockDim.x * 4){
        uint merged_bin = 0;
        merged_bin += s_Hist[pos];
        merged_bin += s_Hist[pos + 1] << 8;
        merged_bin += s_Hist[pos + 2] << 16;
        merged_bin += s_Hist[pos + 3] << 24;
        d_result[pos/4] = merged_bin;
        __syncthreads();
    }
    
}

void opt_2dhisto(uint* d_result, uint* d_data, int dataN,int BIN_COUNT)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
       dim3 blockSize = (1,1,32);
       dim3 gridSize = (1,1,1);
       histogramKernel<<<gridSize,blockSize>>>(d_result, d_data, dataN,BIN_COUNT);
      //for(int i = 0; i < 256; i++) *(d_result + i) = i;
}

/* Include below the implementation of any other functions you need */
uint32_t* allocateInputOnDevice(uint32_t** hostInput,int height, int width){
    //1D Alloc/*
    uint32_t* pointer = *hostInput;
    cudaMalloc((void**)&pointer,4*4096*4096);
    cudaMemcpy(pointer,*hostInput,sizeof(uint32_t)*4096*4096,cudaMemcpyHostToDevice);
    return pointer;
    //uint32_t** rowPointers; //Empty 2DArray
    
    /*cudaMalloc((void**)&rowPointers,sizeof(uint32_t*)*INPUT_HEIGHT); //Create 2d Array on GPU

    uint32_t** pointerArray = (uint32_t**) calloc(INPUT_HEIGHT, sizeof(void*)); // Create host array for row pointer
    
    for (int i = 0 ; i < height;i++){
        cudaMalloc((void**)&pointerArray[i], sizeof(uint32_t) * 4096);//Create actual row vector
        //cudaMemcpy(pointerArray[i], hostInput[i], sizeof(uint32_t) * INPUT_WIDTH, cudaMemcpyHostToDevice);
        std::cout<<"good"<<i<<std::endl;
        std::cout<<hostInput[i][0]<<std::endl;
    }
    cudaMemcpy(rowPointers, pointerArray, sizeof(uint32_t*)*INPUT_HEIGHT, cudaMemcpyHostToDevice);

    free(pointerArray);
*/
    //return rowPointers;
}

uint8_t* allocateHistogramOnDevice(uint8_t** hostHisto, int height, int width){
    uint8_t* deviceHisto = *hostHisto;
    cudaMalloc((void**)&deviceHisto,height*width*sizeof(uint8_t));
    cudaMemcpy(deviceHisto,hostHisto,height*width*sizeof(uint32_t),cudaMemcpyHostToDevice);
    return deviceHisto;
}


void cudaTeardown(uint8_t* deviceHisto, uint8_t*hostHisto, uint32_t* deviceInput){
    cudaDeviceSynchronize();
    cudaMemcpy(hostHisto,deviceHisto,1024*sizeof(uint8_t),cudaMemcpyDeviceToHost);
    cudaFree(deviceHisto);
    
    //for(int i = 0; i<INPUT_HEIGHT; i++){
    //    cudaFree(&deviceInput[i]);
    //}
    cudaFree(deviceInput);
}

