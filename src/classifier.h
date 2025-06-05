#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#define PROCESSED_IMG_SIZE 48
#define VECTOR_LENGTH 3780

namespace fs = std::filesystem;

struct FaceNode {
    int type;
    std::vector<float> vector;
};

class FaceClassifier {
public:
    FaceClassifier();
    ~FaceClassifier();

    void detectFaces(const cv::Mat& gray_frame, std::vector<cv::Rect>& faces);
    std::string classifyFace(const cv::Mat& face_img);
    std::vector<float> vectorizeFace(const cv::Mat& face_img);    
    
private:
    cv::CascadeClassifier face_detector;
  
    float computeDistance(
        const std::vector<float>& face_vector1, 
        const std::vector<float>& face_vector2
    );
    std::string runClassification(std::vector<float> face_vector);    
};

#endif // CLASSIFIER_H