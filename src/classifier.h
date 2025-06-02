#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "forest.h"

#define PROCESSED_IMG_SIZE 48
#define VECTOR_LENGTH 3780

const std::string FACE_TYPES[3] = {"Angry", "Happy", "Sad"};

extern const DecisionTreeNode* const forest[NUM_TREES];

namespace fs = std::filesystem;

class FaceClassifier {
public:
    FaceClassifier();
    ~FaceClassifier();

    void detectFaces(const cv::Mat& gray_frame, std::vector<cv::Rect>& faces);
    std::string classifyFace(const cv::Mat& face_img);
    void vectorizeDataset(
        std::string vectorized_dataset_path = "dataset/vectorized_faces.csv",
        std::string original_dataset_path = "dataset/og/"
    );

private:
    cv::CascadeClassifier face_detector;

    std::vector<float> vectorizeFace(const cv::Mat& face_img);
    std::string runClassification(std::vector<float> face_vector);
    void processFolder(std::string dir_path, uint8_t type_index,std::ofstream& analysis_file);

    void saveFace(const cv::Mat& face_img, uint8_t type_index);
    void saveVectorIntoCsv(
        const std::vector<float>& face_vector, uint8_t type_index, std::ofstream& analysis_file
    );
};

#endif // CLASSIFIER_H