#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto( uint* d_result, uint32_t* d_data, int dataN, int BIN_COUNT);

/* Include below the function headers of any other functions that you implement */
void histogramKernel(uint* d_result, uint* d_data, int dataN,int BIN_COUNT);

uint32_t* allocateInputOnDevice(uint32_t* hostInput);

void copyInputToDevice(uint32_t* deviceInput, uint32_t* hostInput);

uint8_t* allocateHistogramOnDevice(uint8_t* hostHisto);

void copyHistoToDevice(uint8_t* hosthisto, uint8_t* deviceHisto);

void cudaTeardown(uint8_t* deviceHisto, uint8_t*hostHisto,uint32_t* deviceInput);


#endif
