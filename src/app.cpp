/* 
* Script de la app del proyecto.
* Abre una ventana donde se realiza una clasificacion RT
* de las caras que detecta.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "classifier.h"

#define FONT_LABELS cv::FONT_HERSHEY_SIMPLEX
#define FONT_SCALE 0.8
#define FONT_THICKNESS 2
#define FONT_COLOR cv::Scalar(255, 0, 0)

#define RECTANGLE_COLOR cv::Scalar(255, 0, 0)

void test_camara(cv::VideoCapture& cap){
    if (!cap.isOpened()) {
        std::cerr << "Error al abrir la cámara\n";
        exit(1);
    }
}

int main() {
    cv::VideoCapture cap(0);
    test_camara(cap);

    cv::CascadeClassifier face_detector;
    if (!face_detector.load("face_detection/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading Haar\n";
        exit(1);
    }

    IVFClassifier face_classifier;
    cv::Mat frame; 
    cv::Mat gray;
    std::vector<cv::Rect> faces;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
 
        cv::flip(frame, frame, 1);
        face_detector.detectMultiScale(frame, faces);
 
        for (auto& face : faces) {
            cv::Mat face_img = frame(face).clone();
            std::string face_type = face_classifier.classifyFace(face_img);

            cv::rectangle(frame, face, RECTANGLE_COLOR, 2);
            cv::Point textOrg(face.x, face.y - 10);
            cv::putText(
                frame, 
                face_type, 
                textOrg, 
                FONT_LABELS, 
                FONT_SCALE, 
                FONT_COLOR, 
                FONT_THICKNESS
            );
        }

        cv::imshow("Camara RT", frame);
        cv::waitKey(30);
    }
    return 0;  
}