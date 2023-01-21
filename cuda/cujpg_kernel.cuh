#ifndef _CUJPG_KERNEL_CUH_
#define _CUJPG_KERNEL_CUH_

#include <stdio.h>

extern "C"
int YCrCb2RGB(unsigned char *Y_d, unsigned char *Cr_d, unsigned char *Cb_d, int width, int height, int YStep, int CrStep, int CbStep, unsigned char *rgb_d, int nMCUBlocksV, int nMCUBlocksH);


#endif