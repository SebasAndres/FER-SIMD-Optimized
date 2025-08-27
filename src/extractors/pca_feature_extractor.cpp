#include <filesystem>
namespace fs = std::filesystem;

#include "feature_extractor.h"
#include "../face_classifier.h"

PCAFeatureExtractor::~PCAFeatureExtractor() {}

PCAFeatureExtractor::PCAFeatureExtractor(
    const std::string& dataset_path,
    const int num_features
) : FeatureExtractor(dataset_path) {

    FileManager file_manager;

    cv::CascadeClassifier face_detector;
    if (!face_detector.load("detection/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading Haar Cascade XML file\n";
        exit(1);
    }

    // Extract faces from the dataset and convert to 1D
    std::cout << "> Extracting faces from dataset...\n";
    std::vector<std::vector<float>> faces_matrix; 
    SimpleFeatureExtractor simple_extractor("dataset/fer2013");
    file_manager.extractFacesToCSV(
        dataset_path,
        "data/1d_faces",
        faces_matrix,
        face_detector,
        simple_extractor 
    );

    // Compute mean vector of all faces
    std::cout << "> Calculating mean vector...\n";
    this->mean_vector = calculateMeanVector(faces_matrix);
    file_manager.saveVector(
        this->mean_vector, 
        "data/mean_vector.csv"
    );

    // Center every vector using the mean vector
    std::cout << "> Centering vectors...\n";
    centerVectors(faces_matrix, mean_vector);

    // Calculate Covariance matrix
    cv::Mat cov_matrix;
    std::cout << "> Calculating covariance matrix...\n";
    cov_matrix = calculateCovarianceMatrix(faces_matrix);

    // Calculate PCA basis
    std::cout << "> Calculating PCA basis...\n";
    this->pca_basis = calculatePCABasis(
        cov_matrix,
        num_features
    );
    file_manager.saveVectorsToCSV(
        this->pca_basis, 
        "data/pca_basis.csv"
    );

    file_manager.extractFacesToCSV(
        dataset_path,
        "data/projected_faces",
        faces_matrix,
        face_detector,
        *this 
    );
}

std::vector<float> PCAFeatureExtractor::extractFeatures(
    const cv::Mat& image
){
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = image.clone();
    }
    gray.convertTo(gray, CV_8UC1);
    cv::Mat resized_face;
    cv::resize(
        gray, 
        resized_face, 
        cv::Size(PROCESSED_IMG_SIZE, PROCESSED_IMG_SIZE)
    );
    std::vector<float> face_vector = projectInto1D(resized_face);
    if (face_vector.empty()) return std::vector<float>();
    return projectIntoPCA(face_vector, this->pca_basis);
}



