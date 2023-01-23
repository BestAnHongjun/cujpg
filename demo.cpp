#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cujpg.h"
#include "timer.h"


int main()
{
    cuJpgDecoder Decoder("shared/Cat03.jpg");
    Timer decode_timer(std::string("decode"), 5);

    for (int i = 0; i < 100; i++)
    {
        decode_timer.start();
        Decoder.decode(TYPE_BGR);
        cv::Mat res = Decoder.getMatResult();
        decode_timer.end();
    }
    cv::Mat res = Decoder.getMatResult();
    cv::namedWindow("test", 0);
    cv::resizeWindow("test", cv::Size(1280, 720));
    cv::imshow("test", res);
    cv::waitKey(0);
    return 0;
}