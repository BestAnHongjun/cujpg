#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cujpg.h"


int main()
{
    cuJpgDecoder Decoder("shared/test_001.jpg");
    Decoder.decode(TYPE_BGR);
    cv::Mat res = Decoder.getMatResult();
    cv::namedWindow("test", 0);
    cv::resizeWindow("test", cv::Size(1280, 720));
    cv::imshow("test", res);
    cv::waitKey(0);
    return 0;
}