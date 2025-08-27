#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "../linalg.h"
#include <opencv2/opencv.hpp>
#include <vector>

class FeatureExtractor {
public:
    FeatureExtractor(std::string dataset_path);
    ~FeatureExtractor(); 
 
    virtual std::vector<float> extractFeatures(const cv::Mat& image);
    void saveFeatures(const std::string& file_path);

private:
    std::string dataset_path;
};

class PCAFeatureExtractor : public FeatureExtractor {
public:
    PCAFeatureExtractor(
        const std::string& dataset_path,
        const int num_features
    );
    ~PCAFeatureExtractor();

    std::vector<float> extractFeatures(const cv::Mat& image) override;

private:
    std::vector<std::vector<float>> pca_basis;
    std::vector<float> mean_vector;

    void rescanFacesWithPCA(
        const std::string& original_dataset_path, 
        const std::vector<std::vector<float>>& pca_basis
    );
};

// class LBPFeatureExtractor : public FeatureExtractor {
// public:
//     LBPFeatureExtractor(std::string dataset_path);
//     std::vector<float> extractFeatures(const cv::Mat& image) override;
// };

class SimpleFeatureExtractor : public FeatureExtractor {
public:
    SimpleFeatureExtractor(std::string dataset_path);
    std::vector<float> extractFeatures(const cv::Mat& image) override;  
};

#endif // FEATURE_EXTRACTOR_H