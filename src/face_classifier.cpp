#include "face_classifier.h"

FaceClassifier::FaceClassifier(
    FeatureExtractor* feature_extractor,
    EmotionClassifier* emotion_classifier
) {
    /*
    This is the constructor of the FaceClassifier class.
    It initializes the face detector and loads the feature extractor.
    */

    if (!face_detector.load("detection/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading Haar Cascade XML file\n";
        exit(1);
    }
    this->feature_extractor = feature_extractor;
    this->emotion_classifier = emotion_classifier;
}

FaceClassifier::~FaceClassifier() {
    /*
    This is the destructor of the FaceClassifier class.
    It cleans up resources if necessary.
    */
    delete feature_extractor;
    delete emotion_classifier;
}

std::vector<float> FaceClassifier::vectorizeFace(const cv::Mat& face_img) {
    /*
    This function vectorizes a face image using HOG (Histogram of Oriented Gradients).
    It converts the image to grayscale, resizes it to a fixed size, and computes the HOG features.    

    Parameters: 
    - face_img: The input face image in BGR format.
    */
 
    return this->feature_extractor->extractFeatures(face_img);
}


void FaceClassifier::detectFaces(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
    /*
    This function detects faces in a given frame using a Haar Cascade classifier.
    It converts the frame to grayscale and applies the face detection algorithm.

    Parameters:
    - frame: The input image frame in BN format.
    - faces: A vector to store the detected face rectangles.
    */
 
    this->face_detector.detectMultiScale(frame, faces);
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

    return this->emotion_classifier->classify(face_vector);
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