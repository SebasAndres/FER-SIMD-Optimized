/*
This script processes the dataset to compute the media face vector and
PCA basis in order to vectorize any new face
*/


#include "classifier.h"
#include "linalg.h"

#define PROCESSED_IMG_SIZE 48

void extractFacesFromImagesInFolder(
    FaceClassifier& classifier,
    std::string dir_path,
    std::function<std::vector<float>(cv::Mat)> vectorizer_function,
    std::vector<std::vector<float>>& faces_matrix
){
    /*
    This function processes a folder containing images of faces, detects faces in each image,
    vectorizes the detected faces.

    Parameters:
    - dir_path: The path to the directory containing the images.
    - analysis_file: An output file stream to write the vectorized face data.
    */

    std::cout << "*Processing directory: " << dir_path << "\n";

    for (const auto& entry : fs::directory_iterator(dir_path)) {

        // Load image 
        std::string file_path = entry.path().string();
        cv::Mat img = cv::imread(file_path);
        if (img.empty()) {
            std::cerr << "Error loading image: " << file_path << "\n";
            continue;
        }
        
        // Convert image to grayscale if it has 3 channels (BGR)
        cv::Mat gray_img;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
        } else {
            gray_img = img.clone();
        }

        // Detect faces
        std::vector<cv::Rect> faces;
        classifier.detectFaces(gray_img, faces);
 
        // Vectorize each detected face and write to the CSV file
        for (const auto& face : faces) {
            cv::Rect safe_face = face & cv::Rect(0, 0, gray_img.cols, gray_img.rows);
            if (safe_face.area() == 0) continue;
            cv::Mat face_img = gray_img(safe_face).clone();
            std::vector<float> face_vector = vectorizer_function(face_img);
            if (face_vector.empty()) continue;
            faces_matrix.push_back(face_vector);
        }
    }
}


std::vector<float> simpleProjectionVectorizer(cv::Mat face_img){
    cv::Mat resized_face;
    cv::resize(
        face_img, 
        resized_face, 
        cv::Size(PROCESSED_IMG_SIZE, PROCESSED_IMG_SIZE)
    );
    std::vector<float> face_vector = projectInto1D(resized_face);
    return face_vector;
}


void saveVectorsToCSV(
    const std::vector<std::vector<float>>& vectors, 
    const std::string& file_path
) {
    /*
    This function saves a set of vectors to a CSV file.
    Parameters:
    - vectors: A vector of vectors to be saved.
    - file_path: The path to the output CSV file.
    */
    
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << file_path << "\n";
        return;
    }

    for (const auto& vec : vectors) {
        for (size_t i = 0; i < vec.size(); ++i) {
            file << vec[i];
            if (i < vec.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}
 

void collectFaces1D(
    std::string original_dataset_path, 
    std::vector<std::vector<float>>& faces_matrix
) {   
    /*
    This function generates a dataset of vectorized face images from the specified directories.
    It processes images from every subdirectory, which represents a face type.
    */

    std::string faces_dir = "data/faces";

    // Ensure the directory exists before writing files
    if (!fs::exists(faces_dir)) {
        fs::create_directories(faces_dir);
    }

    // Insert datasets for each face type
    FaceClassifier classifier;
    for (const auto& entry : fs::directory_iterator(original_dataset_path)) {
        if (entry.is_directory()) {
            extractFacesFromImagesInFolder(
                classifier, 
                entry.path().string(), 
                simpleProjectionVectorizer,
                faces_matrix
            );
            std::string name = entry.path().filename().string();
            std::string output_csv = faces_dir + "/" + name + "_faces.csv";           
            saveVectorsToCSV(faces_matrix, output_csv);        
        }
    }
}


void saveMeanVector(
    const std::vector<float>& mean_vector, 
    const std::string& file_path
) {
    /*
    This function saves the mean vector to a CSV file.
    Parameters:
    - mean_vector: The mean vector to be saved.
    - file_path: The path to the output CSV file.
    */
    
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << file_path << "\n";
        return;
    }

    for (size_t i = 0; i < mean_vector.size(); ++i) {
        file << mean_vector[i];
        if (i < mean_vector.size() - 1) {
            file << ",";
        }
    }
    file.close();
}


void rescanFacesWithPCA(
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


int main(){
   
    std::cout << "> Generating dataset...\n";
    std::vector<std::vector<float>> faces_matrix;
    collectFaces1D("dataset/fer2013", faces_matrix);    
    
    std::cout << "> Calculating mean vector...\n";
    std::vector<float> mean_vector = calculateMeanVector(faces_matrix);

    std::cout << "> Saving mean vector to 'data/mean_vector.csv'...\n";
    saveMeanVector(mean_vector, "data/mean_vector.csv"); 

    std::cout << "> Centering vectors..\n";
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
    saveVectorsToCSV(pca_basis, "data/pca_basis.csv");

    // Project the one-dimensional faces into PCA space
    rescanFacesWithPCA("dataset/fer2013", pca_basis);
    
    return 0;
}