/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#include "cujpg.h"
using namespace cujpg;


// defined in cujpg_utils.cpp
int DivUp(int x, int d)
{
    return (x + d - 1) / d;
}

uint8_t cuJpgDecoder::nextMarker(const uint8_t* pData, int &nPos, int nLength)
{
    uint8_t c = pData[nPos++];

    do
    {
        while (c != 0xffu && nPos < nLength)
        {
            c = pData[nPos++];
        }

        if (nPos >= nLength)
            return -1;
        
        c = pData[nPos++];
    }

    while (c == 0 || c == 0x0ffu);

    return c;
}

// defined in cujpg_quantization.cpp