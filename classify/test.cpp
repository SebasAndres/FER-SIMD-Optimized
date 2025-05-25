#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cout << "Error al cargar el modelo Haar" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread("data/Happy/1HEoLBLidT2u4mhJ0oiDgig.png");
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces);

    for (auto& face : faces) {
        cv::rectangle(image, face, cv::Scalar(255, 0, 0), 2);

        // Recortar el rostro
        cv::Mat face_img = gray(face).clone();

        // Mostrar
        cv::imshow("Rostro", face_img);
        cv::waitKey(0);
    }

    cv::imshow("Rostros", image);
    cv::waitKey(0);
    return 0;
}