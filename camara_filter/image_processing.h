#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <cstdint>

extern "C" {
    void apply_filter_asm(unsigned char* data, uint32_t width, uint32_t height);
}

// Wrapper C++ para usar con cv::Mat
inline void apply_filter(cv::Mat& image) {
    apply_filter_asm(image.data, image.cols, image.rows);
}

void apply_filter_c(cv::Mat& image);

#endif // IMAGE_PROCESSING_H