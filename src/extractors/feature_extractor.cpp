#include "feature_extractor.h"

std::vector<float> FeatureExtractor::extractFeatures(const cv::Mat& image) {
    // Método base, puede lanzar una excepción si se llama directamente
    throw std::runtime_error("FeatureExtractor::extractFeatures debe ser implementado por una subclase.");
}