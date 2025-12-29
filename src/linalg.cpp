#include "linalg.h"
#include <cstdlib>

// Distancia euclideana --------------------------
float euclideanDistanceC(
    const float* vec1,
    const float* vec2,
    size_t length
){
    float distance = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

float euclideanDistance(
    const float* vec1,
    const float* vec2,
    size_t length
){
    #if USE_ASM_IMP
        return euclideanDistanceASM(vec1, vec2, length);
    #else
        return euclideanDistanceC(vec1, vec2, length);
    #endif
}

// Centrado de vectores -------------------------------
float* centerVectorC(
    const float* vector,
    const float* mean_vector,
    size_t dim
){
    float* result = (float*)malloc(dim * sizeof(float));
    for (size_t i = 0; i < dim; ++i)
        result[i] = vector[i] - mean_vector[i];
    return result;
}

float* centerVector(
    const float* vector,
    const float* mean_vector,
    size_t dim
){
    #if USE_ASM_IMP
        return centerVectorASM(vector, mean_vector, dim);
    #else
        return centerVectorC(vector, mean_vector, dim);
    #endif
}

float** centerVectors(
    float** vectors,
    const float* mean_vector,
    size_t num_vectors,
    size_t vector_dim
){
    float** result = (float**)malloc(num_vectors * sizeof(float*));
    for (size_t i = 0; i < num_vectors; ++i)
        result[i] = centerVector(vectors[i], mean_vector, vector_dim);
    return result;
}

// Calcular vector medio ----------------------------
float* calculateMeanVectorC(
    float** vectors,
    size_t num_vectors,
    size_t vector_dim
){
    float* mean_vector = (float*)malloc(vector_dim * sizeof(float));
    for (size_t j = 0; j < vector_dim; j++) {
        mean_vector[j] = 0.0f;
    }

    for (size_t i = 0; i < num_vectors; i++) {
        for (size_t j = 0; j < vector_dim; j++) {
            mean_vector[j] += vectors[i][j];
        }
    }

    float inv_n = 1.0f / (float)num_vectors;
    for (size_t j = 0; j < vector_dim; j++) {
        mean_vector[j] *= inv_n;
    }

    return mean_vector;
}

float* calculateMeanVector(
    float** vectors,
    size_t num_vectors,
    size_t vector_dim
){
    #if USE_ASM_IMP
        return calculateMeanVectorASM(vectors, num_vectors, vector_dim);
    #else
        return calculateMeanVectorC(vectors, num_vectors, vector_dim);
    #endif
}

// PCA ---------------------------------------

float dotProductC(
    const float* vec1,
    const float* vec2,
    size_t length
){
    float result = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

float dotProduct(
    const float* vec1,
    const float* vec2,
    size_t length
){
    #if USE_ASM_IMP
        return dotProductASM(vec1, vec2, length);
    #else
        return dotProductC(vec1, vec2, length);
    #endif
}

// Proyección PCA: result = pca_basis @ vector
float* projectIntoPCAC(
    const float* vector,
    float** pca_basis,
    size_t num_components,
    size_t vector_dim
){
    float* result = (float*)malloc(num_components * sizeof(float));
    for (size_t i = 0; i < num_components; ++i) {
        result[i] = dotProductC(pca_basis[i], vector, vector_dim);
    }
    return result;
}

float* projectIntoPCA(
    const float* vector,
    float** pca_basis,
    size_t num_components,
    size_t vector_dim
){
    float* result = (float*)malloc(num_components * sizeof(float));
    for (size_t i = 0; i < num_components; ++i) {
        result[i] = dotProduct(pca_basis[i], vector, vector_dim);
    }
    return result;
}

// Matriz de covarianza: C = (1/n) * (X^T * X)
// Retorna matriz aplanada (row-major) de m x m
float* calculateCovarianceMatrixC(
    float** vectors,
    size_t num_vectors,
    size_t vector_dim
){
    size_t matrix_size = vector_dim * vector_dim;
    float* cov_matrix = (float*)malloc(matrix_size * sizeof(float));

    for (size_t i = 0; i < matrix_size; ++i) {
        cov_matrix[i] = 0.0f;
    }

    // Acumulo productos externos
    for (size_t k = 0; k < num_vectors; ++k) {
        for (size_t i = 0; i < vector_dim; ++i) {
            for (size_t j = 0; j < vector_dim; ++j) {
                cov_matrix[i * vector_dim + j] += vectors[k][i] * vectors[k][j];
            }
        }
    }

    // Divido por n
    float inv_n = 1.0f / (float)num_vectors;
    for (size_t i = 0; i < matrix_size; ++i) {
        cov_matrix[i] *= inv_n;
    }

    return cov_matrix;
}

// Wrapper que retorna cv::Mat para compatibilidad con cv::eigen
cv::Mat calculateCovarianceMatrix(
    float** vectors,
    size_t num_vectors,
    size_t vector_dim
){
    float* cov_data = calculateCovarianceMatrixC(vectors, num_vectors, vector_dim);

    // Crear cv::Mat y copiar datos
    cv::Mat cov_matrix(vector_dim, vector_dim, CV_32F);
    memcpy(cov_matrix.data, cov_data, vector_dim * vector_dim * sizeof(float));

    free(cov_data);
    return cov_matrix;
}

float** calculatePCABasis(
    const cv::Mat& covariance_matrix,
    size_t num_components,
    size_t* out_vector_dim
){
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covariance_matrix, eigenvalues, eigenvectors);

    *out_vector_dim = eigenvectors.rows;

    float** pca_basis = (float**)malloc(num_components * sizeof(float*));
    for (size_t i = 0; i < num_components; ++i) {
        pca_basis[i] = (float*)malloc(eigenvectors.rows * sizeof(float));
        for (int j = 0; j < eigenvectors.rows; ++j) {
            pca_basis[i][j] = eigenvectors.at<float>(j, static_cast<int>(i));
        }
    }

    return pca_basis;
}

// -------------------------------------------
float* projectInto1D(const cv::Mat& image, size_t* out_size) {
    CV_Assert(image.type() == CV_8UC1);
    size_t size = image.rows * image.cols;
    *out_size = size;

    float* vector = (float*)malloc(size * sizeof(float));
    size_t idx = 0;
    for (int i = 0; i < image.rows; ++i) {
        const uchar* rowPtr = image.ptr<uchar>(i);
        for (int j = 0; j < image.cols; ++j) {
            vector[idx++] = static_cast<float>(rowPtr[j]);
        }
    }
    return vector;
}

float* extractHOG(const cv::Mat& image, size_t* out_size) {
    cv::HOGDescriptor hog(
        cv::Size(IMG_SIZE, IMG_SIZE),
        cv::Size(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
        cv::Size(HOG_BLOCK_STRIDE, HOG_BLOCK_STRIDE),
        cv::Size(HOG_CELL_SIZE, HOG_CELL_SIZE),
        HOG_NBINS
    );

    cv::Mat resized;
    if (image.cols != IMG_SIZE || image.rows != IMG_SIZE)
        cv::resize(image, resized, cv::Size(IMG_SIZE, IMG_SIZE));
    else
        resized = image;

    cv::Mat gray;
    if (resized.channels() == 3)
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    else
        gray = resized;

    std::vector<float> descriptors;
    hog.compute(gray, descriptors);

    *out_size = descriptors.size();
    float* result = (float*)malloc(descriptors.size() * sizeof(float));
    memcpy(result, descriptors.data(), descriptors.size() * sizeof(float));
    return result;
}

