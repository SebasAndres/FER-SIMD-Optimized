#ifndef FACE_CLASSIFIER_H
#define FACE_CLASSIFIER_HCLASSIFIER_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "file_manager.h"
#include "classifiers/classifier.h"
#include "extractors/feature_extractor.h"
#include "linalg.h"

#define PROCESSED_IMG_SIZE 48

struct FaceNode {
    int type;
    std::vector<float> vector;
};

class FaceClassifier {
public:
    FaceClassifier(
        FeatureExtractor* feature_extractor,
        EmotionClassifier* emotion_classifier
    );
    ~FaceClassifier(); 
    
    void detectFaces(
        const cv::Mat& gray_frame, 
        std::vector<cv::Rect>& faces
    );
    std::string classifyFace(const cv::Mat& face_img);
    std::vector<float> vectorizeFace(const cv::Mat& face_img);    
    
private:
    cv::CascadeClassifier face_detector;  
    FeatureExtractor* feature_extractor;    
    EmotionClassifier* emotion_classifier; 

    std::string runClassification(std::vector<float> face_vector);    
};

#endif 