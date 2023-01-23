/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-20
*/
#include <npp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nppi_compression_functions.h>

#include "cujpg_config.h"
#include "Endianess.h"
#include "cujpg_kernel.cuh"


__global__ void YCrCb2BGRConver(unsigned char *Y_d, unsigned char *Cr_d, unsigned char *Cb_d, unsigned char *bgr_d, int width, int height, int YStep, int CrStep, int CbStep, int nMCUBlocksV, int nMCUBlocksH)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int cols = blockIdx.x*blockDim.x + threadIdx.x;

	if (row >= height)
	{
		return;
	}
	if (cols >= width)
	{
		return;
	}

	int Y = Y_d[row * YStep + cols];
	int U = Cr_d[row / nMCUBlocksH * CrStep + cols / nMCUBlocksV] - 128;
	int V = Cb_d[row / nMCUBlocksH * CbStep + cols / nMCUBlocksV] - 128;

	int B = Y + U + ((U * 198) >> 8);
	int G = Y - (((U * 88) >> 8) + ((V * 183) >> 8));
	int R = Y + V + ((V * 103) >> 8);

	if (B < 0) B = 0;
	if (G < 0) G = 0;
	if (R < 0) R = 0;

	B = B & 0xff;
	G = G & 0xff;
	R = R & 0xff;
    
	bgr_d[row * width * 3 + cols * 3 + 0] = (unsigned char)B;
	bgr_d[row * width * 3 + cols * 3 + 1] = (unsigned char)G;
	bgr_d[row * width * 3 + cols * 3 + 2] = (unsigned char)R;
}

__global__ void YCrCb2RGBConver(unsigned char *Y_d, unsigned char *Cr_d, unsigned char *Cb_d, unsigned char *rgb_d, int width, int height, int YStep, int CrStep, int CbStep, int nMCUBlocksV, int nMCUBlocksH)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int cols = blockIdx.x*blockDim.x + threadIdx.x;

	if (row >= height)
	{
		return;
	}
	if (cols >= width)
	{
		return;
	}

	int Y = Y_d[row * YStep + cols];
	int U = Cr_d[row / nMCUBlocksH * CrStep + cols / nMCUBlocksV] - 128;
	int V = Cb_d[row / nMCUBlocksH * CbStep + cols / nMCUBlocksV] - 128;

	int B = Y + U + ((U * 198) >> 8);
	int G = Y - (((U * 88) >> 8) + ((V * 183) >> 8));
	int R = Y + V + ((V * 103) >> 8);

	if (B < 0) B = 0;
	if (G < 0) G = 0;
	if (R < 0) R = 0;

	B = B & 0xff;
	G = G & 0xff;
	R = R & 0xff;
    
	rgb_d[row * width * 3 + cols * 3 + 0] = (unsigned char)R;
	rgb_d[row * width * 3 + cols * 3 + 1] = (unsigned char)G;
	rgb_d[row * width * 3 + cols * 3 + 2] = (unsigned char)B;
}

__global__ void YCrCb2GrayConver(unsigned char *Y_d, unsigned char *Cr_d, unsigned char *Cb_d, unsigned char *gray_d, int width, int height, int YStep, int CrStep, int CbStep, int nMCUBlocksV, int nMCUBlocksH)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int cols = blockIdx.x*blockDim.x + threadIdx.x;

	if (row >= height)
	{
		return;
	}
	if (cols >= width)
	{
		return;
	}

	gray_d[row * width + cols] = Y_d[row * YStep + cols];
}

int YCrCb2BGR(uint8_t *Y_d, uint8_t *Cr_d, uint8_t *Cb_d, 
	int width, int height, int YStep, int CrStep, int CbStep, 
	uint8_t *res_d, int nMCUBlocksV, int nMCUBlocksH)
{
	dim3 threads(32, 32);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	
    YCrCb2BGRConver <<<blocks, threads>>>(Y_d, Cr_d, Cb_d, res_d, width, height, YStep, CrStep, CbStep, nMCUBlocksV, nMCUBlocksH);
	CHECK(cudaDeviceSynchronize());
	return 0;
}

int YCrCb2RGB(uint8_t *Y_d, uint8_t *Cr_d, uint8_t *Cb_d, 
	int width, int height, int YStep, int CrStep, int CbStep, 
	uint8_t *res_d, int nMCUBlocksV, int nMCUBlocksH)
{
	dim3 threads(32, 32);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	
    YCrCb2RGBConver <<<blocks, threads>>>(Y_d, Cr_d, Cb_d, res_d, width, height, YStep, CrStep, CbStep, nMCUBlocksV, nMCUBlocksH);
	CHECK(cudaDeviceSynchronize());
	return 0;
}

int YCrCb2Gray(uint8_t *Y_d, uint8_t *Cr_d, uint8_t *Cb_d, 
	int width, int height, int YStep, int CrStep, int CbStep, 
	uint8_t *res_d, int nMCUBlocksV, int nMCUBlocksH)
{
	dim3 threads(32, 32);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	
    YCrCb2GrayConver <<<blocks, threads>>>(Y_d, Cr_d, Cb_d, res_d, width, height, YStep, CrStep, CbStep, nMCUBlocksV, nMCUBlocksH);
	CHECK(cudaDeviceSynchronize());
	return 0;
}