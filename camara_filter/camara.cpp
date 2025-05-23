#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_processing.h"

#define FRAMES_DELAY 30

int main() {

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error al abrir la cámara\n";
        return 1;
    }

    cv::Mat frame;
 
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        apply_filter_c(frame);

        cv::imshow("Camara", frame);
        cv::waitKey(FRAMES_DELAY);
    }
    return 0;
}

