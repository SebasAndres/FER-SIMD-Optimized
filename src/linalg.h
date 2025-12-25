#ifndef LINALG_H
#define LINALG_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

// Función implementada en assembly (linalg_asm.asm)
extern "C" {
    float euclideanDistanceASM(const float* vec1, const float* vec2, size_t length);
}

float euclideanDistance(
    const std::vector<float>& face_vector1,
    const std::vector<float>& face_vector2,
    const size_t VECTOR_LENGTH = 3780
); 

std::vector<float> projectInto1D(const cv::Mat& image);

std::vector<float> projectIntoPCA(
    const std::vector<float>& vector, 
    const std::vector<std::vector<float>>& pca_basis
);

std::vector<float> centerVector(
    const std::vector<float>& vector, 
    const std::vector<float>& mean_vector
);

std::vector<float> calculateMeanVector(
    const std::vector<std::vector<float>>& vectors
);

std::vector<std::vector<float>> centerVectors(
    const std::vector<std::vector<float>>& vectors, 
    const std::vector<float>& mean_vector
);

std::vector<std::vector<float>> calculatePCABasis(
    const cv::Mat& covariance_matrix, 
    const size_t num_components = 100
);

cv::Mat calculateCovarianceMatrix(
    const std::vector<std::vector<float>>& vectors
);

#endif