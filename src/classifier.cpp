#include "classifier.h"


FaceClassifier::FaceClassifier() {
    if (!face_detector.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error al cargar el modelo Haar\n";
        exit(1);
    }
}


FaceClassifier::~FaceClassifier() {
}


std::vector<float> FaceClassifier::vectorizeFace(const cv::Mat& face_img) {
    /*
    This function vectorizes a face image using HOG (Histogram of Oriented Gradients).
    It converts the image to grayscale, resizes it to a fixed size, and computes the HOG features.    

    Parameters: 
    - face_img: The input face image in BGR format.
    */

    std::vector<float> vector;

    // Convert the image to grayscale (if it's not already)
    cv::Mat bn_face;
    cvtColor(face_img, bn_face, cv::COLOR_BGR2GRAY);

    // Resize the image to a fixed size
    cv::resize(bn_face, bn_face, cv::Size(64, 128));

    // Extract feactures
    

    return vector;
}


void FaceClassifier::detectFaces(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
    /*
    This function detects faces in a given frame using a Haar Cascade classifier.
    It converts the frame to grayscale and applies the face detection algorithm.

    Parameters:
    - frame: The input image frame in BGR format.
    - faces: A vector to store the detected face rectangles.
    */
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    face_detector.detectMultiScale(gray_frame, faces);
}


float FaceClassifier::computeDistance(
    const std::vector<float>& face_vector1, 
    const std::vector<float>& face_vector2
) {
    /*
    This function computes the Euclidean distance between two face vectors.
    Parameters:
    - face_vector1: The first face vector.
    - face_vector2: The second face vector.
    Returns:
    - The Euclidean distance between the two vectors.
    */
    
    float distance = 0.0f;
    for (size_t i = 0; i < VECTOR_LENGTH; ++i) {
        float diff = face_vector1[i] - face_vector2[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}


std::string FaceClassifier::runClassification(std::vector<float> face_vector) {
    /*
    This function runs the classification of a face vector using the KNN algorithm.
    This could run parallel to speed up the classification process using SIMD.

    Parameters:
    - face_vector: A vector of features extracted from the face image.
    Returns:
    - The predicted face type as a string.
    */
    return "Happy";
}


std::string FaceClassifier::classifyFace(const cv::Mat& face_img) {
    /*
    This function classifies a face image by first vectorizing it and then running the classification.
    Parameters:
    - face_img: The input face image in BGR format.
    Returns:
    - The predicted face type as a string.
    */

    std::vector<float> face_vector = vectorizeFace(face_img);
    std::string face_type = runClassification(face_vector);
    return face_type;
}