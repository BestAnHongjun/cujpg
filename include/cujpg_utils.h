/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#ifndef __CUJPG_UTILS_H__
#define __CUJPG_UTILS_H__

#include "Endianess.h"


int DivUp(int x, int d);

template<typename T>
T readAndAdvance(const unsigned char *&pData)
{
    T nElement = readBigEndian<T>(pData);
    pData += sizeof(T);
    return nElement;
}

#endif // __CUJPG_UTILS_H__
