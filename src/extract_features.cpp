/*
This script processes the dataset to compute the media face vector and
PCA basis in order to vectorize any new face
*/

#include "face_classifier.h"
#include "linalg.h"

#define PROCESSED_IMG_SIZE 48

void saveVector(
    const std::vector<float>& vector, 
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
 
void extractFacesFromImagesInFolder(
    cv::CascadeClassifier& face_detector,
    FeatureExtractor& feature_extractor,
    std::string dir_path,
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
        face_detector.detectMultiScale(gray_img, faces);
 
        // Vectorize each detected face and write to the CSV file
        for (const auto& face : faces) {
            // Ensure the face rectangle is within the image bounds
            cv::Rect safe_face = face & cv::Rect(0, 0, gray_img.cols, gray_img.rows);
            if (safe_face.area() == 0) continue;

            // Extract the face image and vectorize it
            cv::Mat face_img = gray_img(safe_face).clone();
            std::vector<float> face_vector = face_extractor.extract_features(face_img);
            if (face_vector.empty()) continue;
            faces_matrix.push_back(face_vector);
        }
    }
}

void extractFacesToCSV(
    std::string original_dataset_path, 
    std::string output_dir,
    std::vector<std::vector<float>>& faces_matrix,
    cv::CascadeClassifier& face_detector,
    FeatureExtractor& face_extractor,
) {   
    /*
    This function generates a dataset of vectorized face images from the specified directories.
    It processes images from every subdirectory, which represents a face type.
    */

    // Ensure the directory exists before writing files
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    // Insert datasets for each face type
    for (const auto& entry : fs::directory_iterator(original_dataset_path)) {
        if (entry.is_directory()) {
            extractFacesFromImagesInFolder(
                face_detector, 
                feature_extractor,
                entry.path().string(), 
                faces_matrix
            );
            std::string name = entry.path().filename().string();
            std::string output_csv = output_dir + "/" + name + "_faces.csv";           
            saveVectorsToCSV(faces_matrix, output_csv);        
        }
    }
}

int main(){
    std::cout << "[1] Initialize face detector\n";
    cv::CascadeClassifier face_detector;
    if (!face_detector.load("data/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading Haar Cascade XML file\n";
        return -1;
    }

    std::cout << "[2] Initialize feature extractor\n";
    PCAFeatureExtractor pca_extractor("dataset/fer2013");

    std::cout << "[3] Extract features of dataset\n";
    extractFacesToCSV(
        "dataset/fer2013",
        "data/extracted_faces",
        face_detector,
        pca_extractor,
    );

    return 0;
}