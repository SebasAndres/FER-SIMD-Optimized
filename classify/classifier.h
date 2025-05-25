#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#define PROCESSED_IMG_SIZE 100
#define VECTOR_LENGTH (PROCESSED_IMG_SIZE * PROCESSED_IMG_SIZE * 3)

const std::string FACE_TYPES[3] = {"Enojado", "Feliz", "Triste"};

namespace fs = std::filesystem;

class FaceClassifier {
public:
    FaceClassifier();
    ~FaceClassifier();

    void detect_faces(const cv::Mat& gray_frame, std::vector<cv::Rect>& faces);
    std::string classify_face(const cv::Mat& face_img);
    void train();

private:
    cv::CascadeClassifier face_detector;
    cv::Mat gray_frame;

    float centroids[3][VECTOR_LENGTH];

    std::string run_classification(const float* face_vector);
    float* preprocess_face(const cv::Mat& face_img);
    void calculate_centroid(std::string dir_path, uint8_t type_index, std::ofstream& analysis_file);

    void save_centroids(const std::string& filename);
    bool load_centroids(const std::string& filename);

};

#endif // CLASSIFIER_H