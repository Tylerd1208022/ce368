#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto( uint8_t* d_result, uint32_t* d_data, int dataN, int BIN_COUNT);

/* Include below the function headers of any other functions that you implement */
void histogramKernel(uint* d_result, uint* d_data, int dataN,int BIN_COUNT);

uint8_t* initHisto(uint8_t* hostHisto);

uint32_t* initInput(uint32_t* hostHisto);

void cudaTeardown(uint8_t* deviceHisto, uint8_t*hostHisto,uint32_t* deviceInput);

#endif
