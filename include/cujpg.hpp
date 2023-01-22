/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#ifndef __CUJPG_HPP__
#define __CUJPG_HPP__

#include <cstdlib>
#include <unistd.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>


enum imgType
{
    TYPE_RGB     = 0,
    TYPR_BGR     = 1,
    TYPE_GRAY    = 2
};


class cuJpgDecoder
{
private:
    uint8_t* pJpegData;
    int64_t nDataLength;

public:
    cuJpgDecoder(){}
    cuJpgDecoder(const uint8_t* pJpegData, int64_t nDataLength);
    ~cuJpgDecoder();

    void init(const uint8_t* pJpegData, int64_t nDataLength);
    void decode(imgType type);
    uint8_t* getBufferResult();
    cv::Mat getMatResult();
};

#endif // __CUJPG_HPP__
