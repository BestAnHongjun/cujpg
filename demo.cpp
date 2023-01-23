/*
* Copyright © Coder.AN.  All rights reserved.
*
* an.hongjun@foxmail.com
* https://github.com/BestAnHongjun
* 2023-01-22
*/
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "jpg.h"
#include "cujpg.h"
#include "timer.h"


cujpg::imgBuffer* load_jpg(const char* jpgFileName)
{
    std::ifstream stream(jpgFileName, std::ifstream::binary);
    if (!stream.good())
    {
        std::cerr << "Error! Can not open jpg file:" << jpgFileName << std::endl;
        exit(-1);
    }

    stream.seekg(0, std::ios::end);
    int32_t nDataLength = (int64_t)stream.tellg();
    stream.seekg(0, std::ios::beg);

    uint8_t* pJpgData = new uint8_t[nDataLength];
    stream.read((char*)pJpgData, nDataLength);

    cujpg::imgBuffer* res = new cujpg::imgBuffer;
    res->start = pJpgData;
    res->length = nDataLength;

    return res;
}

void show_image(const char* windowName, cv::Mat& mat)
{
    cv::namedWindow(windowName, 0);
    cv::resizeWindow(windowName, cv::Size(1280, 720));
    cv::imshow(windowName, mat);
    cv::waitKey(0);
}


int main()
{
    cujpg::imgBuffer* jpg_buffer = load_jpg("shared/test001.jpg");
    cujpg::cuJpgDecoder Decoder(cujpg::GPU);
    
    cv::Mat res;
    cujpg::imgBuffer* res_buffer;
    cujpg::imgBuffer* dst_buffer;
    Timer cuBgrTimer(std::string("4K JPEG to BGR (cuda)"), 5);
    Timer rgbTimer(std::string("4K JPEG to RGB (cpu) "), 5);
    Timer cuRgbTimer(std::string("4K JPEG to RGB (cuda)"), 5);
    Timer cuGrayTimer(std::string("4K JPEG to GRAY(cuda)"), 5);
    

    // Test BGR-cuda decode
    for (int i = 0; i < 100; i++)
    {
        cuBgrTimer.start();
        Decoder.setSrcBuffer(jpg_buffer);
        Decoder.decode(cujpg::TYPE_BGR);
        res_buffer = Decoder.getBufferResult();
        cuBgrTimer.end();
    }
    res = Decoder.getMatResult();
    show_image("BGR-cuda", res);

    // Test RGB decode
    dst_buffer = new cujpg::imgBuffer;
    for (int i = 0; i < 100; i++)
    {
        rgbTimer.start();
        JPG2BGR(jpg_buffer, dst_buffer);
        rgbTimer.end();
        delete[] dst_buffer->start;
    }
    res = cv::Mat(dst_buffer->height, dst_buffer->width, CV_8UC3, (void*)dst_buffer->start);
    show_image("RGB", res);
    delete dst_buffer;

    // Test RGB-cuda decode
    for (int i = 0; i < 100; i++)
    {
        cuRgbTimer.start();
        Decoder.setSrcBuffer(jpg_buffer);
        Decoder.decode(cujpg::TYPE_RGB);
        res_buffer = Decoder.getBufferResult();
        cuRgbTimer.end();
    }
    res = Decoder.getMatResult();
    show_image("RGB-cuda", res);

    // Test Gray-cuda decode
    for (int i = 0; i < 100; i++)
    {
        cuGrayTimer.start();
        Decoder.setSrcBuffer(jpg_buffer);
        Decoder.decode(cujpg::TYPE_GRAY);
        res_buffer = Decoder.getBufferResult();
        cuGrayTimer.end();
    }
    res = Decoder.getMatResult();
    show_image("Gray-cuda", res);

    return 0;
}