#include "linalg.h"

float euclideanDistance(
    const std::vector<float>& face_vector1,
    const std::vector<float>& face_vector2,
    const size_t VECTOR_LENGTH
){
    /*
    This function computes the Euclidean distance between two face vectors.
    Parameters:
    - face_vector1: The first face vector.
    - face_vector2: The second face vector.
    Returns:
    - The Euclidean distance between the two vectors.
    */
    
    float distance = 0.0f;
    size_t n = face_vector1.size();
    for (size_t i = 0; i < n; ++i) {
        float diff = face_vector1[i] - face_vector2[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}


std::vector<float> projectInto1D(const cv::Mat& image) {
    /*
    This function projects a 2D image into a 1D vector.
    Parameters:
    - image: The input image in grayscale format.
    Returns:
    - A 1D vector representation of the image.
    */
    
    std::vector<float> vector;
    vector.reserve(image.rows * image.cols);

    // Asegúrate de que la imagen es de tipo uchar (CV_8UC1)
    CV_Assert(image.type() == CV_8UC1);

    for (int i = 0; i < image.rows; ++i) {
        const uchar* rowPtr = image.ptr<uchar>(i); 
        for (int j = 0; j < image.cols; ++j) {
            float cell = static_cast<float>(rowPtr[j]); 
            vector.push_back(cell);
        }
    }

    return vector;
}

std::vector<float> projectIntoPCA(
    const std::vector<float>& vector, 
    const std::vector<std::vector<float>>& pca_basis
) {
    /*
    This function projects a vector into PCA space.
    Parameters:
    - vector: The input vector to be projected.
    - pca_basis: The PCA basis vectors.
    Returns:
    - A vector projected into PCA space.
    */
   
   // projected_vector := pcaBasis @ x_norm
   // where pcaBasis is the PCA basis matrix and x_norm is the normalized input vector
 
    std::vector<float> projected_vector;
    size_t k = pca_basis.size();
    
    for (size_t i = 0; i < k; ++i) {
        float projection = 0.0f;
        for (size_t j = 0; j < vector.size(); ++j) {
            projection += pca_basis[i][j] * vector[j];
        }
        projected_vector.push_back(projection);
    }

    return projected_vector;
}


std::vector<float> centerVector(
    const std::vector<float>& vector, 
    const std::vector<float>& mean_vector
){
    /*
    This function subtracts the calculated mean vector from the input vector.
    Parameters:
    - vector: The input vector from which the mean will be subtracted.
    - mean_vector: The mean vector to be subtracted.
    Returns:
    - A new vector with the mean subtracted.
    */

    std::vector<float> result(vector.size());    
    for (size_t i = 0; i < vector.size(); ++i) {
        result[i] = vector[i] - mean_vector[i];
    }

    return result;
}

std::vector<std::vector<float>> centerVectors(
    const std::vector<std::vector<float>>& vectors, 
    const std::vector<float>& mean_vector
){
    /*
    This function centers a set of vectors by subtracting the mean vector from each vector.
    Parameters:
    - vectors: A vector of vectors to be centered.
    - mean_vector: The mean vector to be subtracted from each vector.
    Returns:
    - A new vector of centered vectors.
    */

    std::vector<std::vector<float>> centered_vectors;
    centered_vectors.reserve(vectors.size());

    for (const auto& vec : vectors) {
        centered_vectors.push_back(centerVector(vec, mean_vector));
    }

    return centered_vectors;
}


std::vector<float> calculateMeanVector(
    const std::vector<std::vector<float>>& vectors
) {
    /*
    This function calculates the mean vector from a set of vectors.

    Parameters:
    - vectors: A vector of vectors from which the mean will be calculated.
    Returns:
    - The mean vector.
    */

    std::vector<float> mean_vector(vectors[0].size(), 0.0f);
    
    for (const auto& vec : vectors) {
        for (size_t i = 0; i < vec.size(); ++i) {
            mean_vector[i] += vec[i];
        }
    }
    
    for (auto& value : mean_vector) {
        value /= vectors.size();
    }
    
    return mean_vector;
}

cv::Mat calculateCovarianceMatrix(
    const std::vector<std::vector<float>>& vectors
){
    /*
    This function calculates the covariance matrix from a set of vectors.

                             C = (1/n) * (X^T * X)

    where C is the covariance matrix, n is the number of vectors, and X is the matrix 
    formed by stacking the vectors.

    Parameters:
    - vectors: A vector of vectors from which the covariance matrix will be calculated.
    
    Returns:
    - The covariance matrix as a cv::Mat object.
    */

    size_t n = vectors.size();
    size_t m = vectors[0].size();
    
    cv::Mat covariance_matrix = cv::Mat::zeros(m, m, CV_32F);
    
    for (const auto& vec : vectors) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                covariance_matrix.at<float>(i, j) += vec[i] * vec[j];
            }
        }
    }
    
    covariance_matrix /= n;
    
    return covariance_matrix;
}


std::vector<std::vector<float>> calculatePCABasis(
    const cv::Mat& covariance_matrix, 
    const size_t num_components
) {
    /*
    This function calculates the PCA basis vectors from the covariance matrix.
    Parameters:
    - covariance_matrix: The covariance matrix from which the PCA basis will be calculated.
    - num_components: The number of PCA components to retain.
    
    Returns:
    - A vector of vectors representing the PCA basis.
    */

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covariance_matrix, eigenvalues, eigenvectors);

    std::cout << "--> num_components: " << num_components << "\n";
    
    std::vector<std::vector<float>> pca_basis;
    
    for (int i = 0; i < num_components; ++i) {
        std::vector<float> component;
        for (int j = 0; j < eigenvectors.rows; ++j) {
            component.push_back(eigenvectors.at<float>(j, i));
        }
        pca_basis.push_back(component);
    }
    
    return pca_basis;
}