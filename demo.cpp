#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cujpg.hpp"


unsigned char* load_jpg(const char* input_file, int &nInputLength)
{
    std::ifstream stream(input_file, std::ifstream::binary);
    if (!stream.good())
    {
        return NULL;
    }

    stream.seekg(0, std::ios::end);
    nInputLength = (int)stream.tellg();
    stream.seekg(0, std::ios::beg);

    unsigned char* pJpegData = new unsigned char[nInputLength];
    stream.read((char*)pJpegData, nInputLength);

    return pJpegData;
}


int main()
{
    int nDataLen;
    unsigned char* pJpegData = load_jpg("shared/test_001.jpg", nDataLen);

    if (pJpegData == NULL)
    {
        printf("Read Error!\n");
        delete[] pJpegData;
        return 0;
    }

    int width, height;
    unsigned char* rgbBuffer = jpgToRgb(pJpegData, nDataLen, width, height);
    cv::Mat img = cv::Mat(height, width, CV_8UC3, (void*)rgbBuffer);

    cv::imshow("img", img);
    cv::waitKey(0);

    free(rgbBuffer);
    delete[] pJpegData;
    return 0;
}