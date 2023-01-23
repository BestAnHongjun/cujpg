/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#ifndef _CUJPG_KERNEL_CUH_
#define _CUJPG_KERNEL_CUH_

#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

extern "C"
int YCrCb2BGR(uint8_t *Y_d, uint8_t *Cr_d, uint8_t *Cb_d, 
    int width, int height, int YStep, int CrStep, int CbStep, 
    uint8_t *res_d, int nMCUBlocksV, int nMCUBlocksH);

extern "C"
int YCrCb2RGB(uint8_t *Y_d, uint8_t *Cr_d, uint8_t *Cb_d, 
    int width, int height, int YStep, int CrStep, int CbStep, 
    uint8_t *res_d, int nMCUBlocksV, int nMCUBlocksH);

extern "C"
int YCrCb2Gray(uint8_t *Y_d, uint8_t *Cr_d, uint8_t *Cb_d, 
    int width, int height, int YStep, int CrStep, int CbStep, 
    uint8_t *res_d, int nMCUBlocksV, int nMCUBlocksH);


#endif