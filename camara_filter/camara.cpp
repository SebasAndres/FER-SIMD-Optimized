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

        cv::Mat frame_copy;
        frame.copyTo(frame_copy);
        apply_filter_asm(frame_copy);

        // GaussianBlur(frame, frame, cv::Size(5, 5), 0);

        cv::imshow("Camara", frame_copy);
        cv::waitKey(FRAMES_DELAY);
    }
    return 0;
}

