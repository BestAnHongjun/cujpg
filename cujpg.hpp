/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-20
*/
#ifndef _CUJPG_HPP_
#define _CUJPG_HPP_


#include <npp.h>
#include <cuda_runtime.h>

#include "Endianess.h"
#include "Exceptions.h"
#include "helper_cuda.h"

#include <math.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>


unsigned char* jpgToRgb(unsigned char* pJpegData, int nInputLength, int &pwidth, int &pheight);


#endif  // _CUJPG_HPP_