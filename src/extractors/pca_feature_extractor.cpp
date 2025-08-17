#include "feature_extractor.h"

void PCAFeatureExtractor::rescanFacesWithPCA(
    const std::string& original_dataset_path, 
    const std::vector<std::vector<float>>& pca_basis
) {
    /*
    This function rescans the faces in the original dataset and projects them into PCA space.
    It saves the projected faces in a new directory named "pca_faces".
    
    Parameters:
    - original_dataset_path: The path to the original dataset containing face images.
    - pca_basis: The PCA basis vectors used for projection.
    */
    
    std::string pca_faces_dir = "data/pca_faces";
    fs::create_directory(pca_faces_dir);

    FaceClassifier classifier;     
    for (const auto& entry : fs::directory_iterator(original_dataset_path)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            std::string output_csv = pca_faces_dir + "/" + name + "_faces.csv";           
            std::vector<std::vector<float>> projected_faces;
            extractFacesFromImagesInFolder(
                classifier,
                entry.path().string(), 
                [&](cv::Mat face) {
                    std::vector<float> face_vector = projectInto1D(face);
                    if (face_vector.empty()) return std::vector<float>();
                    return projectIntoPCA(face_vector, pca_basis);
                },
                projected_faces
            );
            saveVectorsToCSV(projected_faces, output_csv);        
        }
    }
}

PCAFeatureExtractor::PCAFeatureExtractor(
    const std::string& dataset_path
) : dataset_path(dataset_path) {
    // Extract faces from the dataset and convert to 1D
    std::cout << "> Extracting faces from dataset...\n";
    std::vector<std::vector<float>> faces_matrix; 
    extractFacesToCSV(
        dataset_path,
        "data/extracted_faces",
        faces_matrix,
        face_detector,
        *SimpleFeatureExtractor("dataset/fer2013")
    );

    // Compute mean vector of all faces
    std::cout << "> Calculating mean vector...\n";
    std::vector<float> mean_vector = calculateMeanVector(faces_matrix);
    saveVector(mean_vector, "data/pca/mean_vector.csv"); 

    // Center every vector using the mean vector
    std::cout << "> Centering vectors...\n";
    centerVectors(faces_matrix, mean_vector);

    // Calculate Covariance matrix
    std::cout << "> Calculating covariance matrix...\n";
    cv::Mat cov_matrix = calculateCovarianceMatrix(faces_matrix);

    // Calculate PCA basis
    std::cout << "> Calculating PCA basis...\n";
    std::vector<std::vector<float>> pca_basis = calculatePCABasis(
        cov_matrix,
        1000
    );

    // Save PCA basis to file
    std::cout << "> Saving pca matrix to 'data/pca_basis.csv'...\n";
    saveVectorsToCSV(pca_basis, "data/pca/pca_basis.csv");

    // Project the one-dimensional faces into PCA space
    rescanFacesWithPCA("dataset/fer2013", pca_basis);
}

std::vector<std::vector<float>> FaceClassifier::loadVectorsFromCsv(const std::string& file_path) {
    /*

    std::vector<std::vector<float>> loadVectorsFromCsv(const std::string& file_path);


    This function loads vectors from a CSV file.
    Parameters:
    - file_path: The path to the CSV file containing vectors.
    
    Returns:
    - A vector of vectors representing the loaded data.
    */

    std::vector<std::vector<float>> vectors;
    std::ifstream file(file_path);

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> vector;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            vector.push_back(std::stof(value));
        }
        vectors.push_back(vector);
    }

    return vectors;
}


std::vector<float> PCAFeatureExtractor::extractFeatures(
    const cv::Mat& image
){
    std::vector<float> face_vector = projectInto1D(image);
    if (face_vector.empty()) return std::vector<float>();
    return projectIntoPCA(face_vector, pca_basis);
}



