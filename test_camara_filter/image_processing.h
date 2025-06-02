#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <cstdint>

extern "C" void apply_filter_asm(cv::Mat& image);

void apply_filter_asm(cv::Mat& image);
void apply_filter_c(cv::Mat& image);

#endif // IMAGE_PROCESSING_H