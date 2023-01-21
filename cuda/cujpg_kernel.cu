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

#include "Endianess.h"

#include "cujpg_kernel.cuh"


__device__ unsigned char judge(int value)
{
	if (value >= 0 && value <= 255)
	{
		return value;
	}
	else if (value>255)
	{
		return 255;
	}
	else
	{
		return 0;
	}
}

__global__ void YCrCb2RGBConver(unsigned char *Y_d, unsigned char *Cr_d, unsigned char *Cb_d, unsigned char *rgb_d, int width, int height, int YStep, int CrStep, int CbStep, int nMCUBlocksV, int nMCUBlocksH)
{
	//int tid = blockIdx.x*blockDim.x + threadIdx.x;
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

	int Y = Y_d[row*YStep + cols];
	int U = Cr_d[row / nMCUBlocksH*CrStep + cols / nMCUBlocksV] - 128;
	int V = Cb_d[row / nMCUBlocksH*CbStep + cols / nMCUBlocksV] - 128;
    int img_step = width * 3;

	rgb_d[row*img_step + cols * 3 + 0] = judge(Y + U + ((U * 198) >> 8));
	rgb_d[row*img_step + cols * 3 + 1] = judge(Y - (((U * 88) >> 8) + ((V * 183) >> 8)));
	rgb_d[row*img_step + cols * 3 + 2] = judge(Y + V + ((V * 103) >> 8));
}

int YCrCb2RGB(unsigned char *Y_d, unsigned char *Cr_d, unsigned char *Cb_d, int width, int height, int YStep, int CrStep, int CbStep, unsigned char *rgb_d, int nMCUBlocksV, int nMCUBlocksH)
{
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	dim3 threads(8, 8);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	
    YCrCb2RGBConver <<<blocks, threads>>>(Y_d, Cr_d, Cb_d, rgb_d, width, height, YStep, CrStep, CbStep, nMCUBlocksV, nMCUBlocksH);
	
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("核函数消耗时间:%f\n", time);
	return 0;
}