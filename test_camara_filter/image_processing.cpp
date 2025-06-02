#include "image_processing.h"

void apply_filter_c(cv::Mat& image) {       
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {    
            cv::Vec3b& pixel = image.at<cv::Vec3b>(y, x);
            uint8_t rojo = pixel[2];
            uint8_t verde = pixel[1];
            uint8_t azul = pixel[0];
            uint8_t luminosidad = (uint8_t)(0.2126 * rojo + 0.7152 * verde + 0.0722 * azul);
            pixel[0] = luminosidad;
            pixel[1] = luminosidad;
            pixel[2] = luminosidad;
        }
    }
}