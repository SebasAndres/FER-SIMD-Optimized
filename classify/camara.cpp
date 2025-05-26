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

    FaceClassifier face_classifier = FaceClassifier();
    face_classifier.train();    

    cv::VideoCapture cap(0);
    test_camara(cap);

    cv::Mat frame; 
    cv::Mat gray;
    std::vector<cv::Rect> faces;

    while (true) {

        // Read from the camera
        cap >> frame;
        
        // Check if the frame is empty
        if (frame.empty()) break;

        // Apply the face detection
        face_classifier.detect_faces(frame, faces);

        for (auto& face : faces) {
            // Crop the face
            cv::Mat face_img = frame(face).clone();

            // Get the face type
            std::string face_type = face_classifier.classify_face(face_img);

            // Draw rectangle around the detected face
            cv::rectangle(frame, face, RECTANGLE_COLOR, 2);

            // Draw text above the rectangle
            cv::Point textOrg(face.x, face.y - 10);
            cv::putText(frame, face_type, textOrg, FONT_LABELS, FONT_SCALE, FONT_COLOR, FONT_THICKNESS);
        }

        cv::imshow("Camara (clasificada)", frame);
        cv::waitKey(30);
    }

    return 0;  
}