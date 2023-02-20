#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto( uint* d_result, uint32_t* d_data, int dataN, int BIN_COUNT);

/* Include below the function headers of any other functions that you implement */

uint32_t* allocateInputOnDevice(uint32_t** hostInput,int height, int width);

uint8_t* allocateHistogramOnDevice(uint8_t** hostHisto,int height, int width);

void cudaTeardown(uint8_t* deviceHisto, uint8_t* hostHisto,uint32_t* deviceInput);


#endif
