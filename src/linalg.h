#ifndef LINALG_H
#define LINALG_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "types.h"
#include "constants.h"

// ============================================
// Funciones de assembly
// ============================================
extern "C" {
    float euclideanDistanceASM(
        const float* vec1,
        const float* vec2,
        size_t length
    );
    float* centerVectorASM(
        const float* vector,
        const float* mean_vector,
        size_t dim
    );
    float* calculateMeanVectorASM(
        float** vectors,
        size_t num_vectors,
        size_t vector_dim
    );
    float dotProductASM(
        const float* vec1,
        const float* vec2,
        size_t length
    );
}

// ============================================
// Distancia euclideana
// ============================================
float euclideanDistance(
    const float* vec1,
    const float* vec2,
    size_t length
);

float euclideanDistanceC(
    const float* vec1,
    const float* vec2,
    size_t length
);

// ============================================
// Centrado de vectores
// ============================================
float* centerVector(
    const float* vector,
    const float* mean_vector,
    size_t dim
);

float* centerVectorC(
    const float* vector,
    const float* mean_vector,
    size_t dim
);

float** centerVectors(
    float** vectors,
    const float* mean_vector,
    size_t num_vectors,
    size_t vector_dim
);

// ============================================
// Vector medio
// ============================================
float* calculateMeanVector(
    float** vectors,
    size_t num_vectors,
    size_t vector_dim
);

float* calculateMeanVectorC(
    float** vectors,
    size_t num_vectors,
    size_t vector_dim
);

// ============================================
// PCA
// ============================================
float dotProduct(
    const float* vec1,
    const float* vec2,
    size_t length
);

float dotProductC(
    const float* vec1,
    const float* vec2,
    size_t length
);

float* projectIntoPCA(
    const float* vector,
    float** pca_basis,
    size_t num_components,
    size_t vector_dim
);

float* projectIntoPCAC(
    const float* vector,
    float** pca_basis,
    size_t num_components,
    size_t vector_dim
);

float** calculatePCABasis(
    const cv::Mat& covariance_matrix,
    size_t num_components,
    size_t* out_vector_dim
);

// ============================================
// Matriz de covarianza
// ============================================
cv::Mat calculateCovarianceMatrix(
    float** vectors,
    size_t num_vectors,
    size_t vector_dim
);

float* calculateCovarianceMatrixC(
    float** vectors,
    size_t num_vectors,
    size_t vector_dim
);

// ============================================
// Extracción de features
// ============================================
float* projectInto1D(const cv::Mat& image, size_t* out_size);

float* extractHOG(const cv::Mat& image, size_t* out_size);

// ============================================
inline void freeVector(float* vec) {
    free(vec);
}

inline void freeMatrix(float** matrix, size_t num_rows) {
    for (size_t i = 0; i < num_rows; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}

#endif
