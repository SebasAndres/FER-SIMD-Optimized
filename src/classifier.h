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

#include "linalg.h"

#define PROCESSED_IMG_SIZE 48

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
    std::vector<float> meanVector;
    std::vector<std::vector<float>> pcaBasis;

    
    std::string runClassification(std::vector<float> face_vector);    
    std::vector<std::vector<float>> loadVectorsFromCsv(const std::string& file_path);
};

#endif // CLASSIFIER_H