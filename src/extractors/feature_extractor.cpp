#include "feature_extractor.h"

FeatureExtractor::FeatureExtractor(std::string dataset_path) {
    this->dataset_path = dataset_path;
}

FeatureExtractor::~FeatureExtractor() {

}

std::vector<float> FeatureExtractor::extractFeatures(const cv::Mat& image) {
}