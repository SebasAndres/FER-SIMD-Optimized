#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>

#include <filesystem>
namespace fs = std::filesystem;

#include "extractors/feature_extractor.h"

class FileManager {
public:
    static std::vector<std::pair<std::string, std::vector<float>>> loadClassifiedVectors(const std::string& dataset_path);
    static void saveVector(
        const std::vector<float>& vector,
        const std::string & file_path
    );
    static void saveVectorsToCSV(
        const std::vector<std::vector<float>>& vectors, 
        const std::string& file_path
    );
    static void extractFacesFromImagesInFolder(
        cv::CascadeClassifier& face_detector,
        FeatureExtractor& feature_extractor,
        std::string dir_path,   
        std::vector<std::vector<float>>& faces_matrix
    );
    static void extractFacesToCSV(
        std::string original_dataset_path,
        std::string output_dir,
        std::vector<std::vector<float>>& faces_matrix,
        cv::CascadeClassifier& face_detector,
        FeatureExtractor& face_extractor
    );
};

#endif