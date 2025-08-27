#include "file_manager.h"

void FileManager::saveVector(
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

    for (size_t i = 0; i < vector.size(); ++i) {
        file << vector[i];
        if (i < vector.size() - 1) {
            file << ",";
        }
    }
    file.close();
}

void FileManager::saveVectorsToCSV(
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
 
void FileManager::extractFacesFromImagesInFolder(
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
            std::vector<float> face_vector = feature_extractor.extractFeatures(face_img);
            if (face_vector.empty()) continue;
            faces_matrix.push_back(face_vector);
        }
    }
}

void FileManager::extractFacesToCSV(
    std::string original_dataset_path, 
    std::string output_dir,
    std::vector<std::vector<float>>& faces_matrix,
    cv::CascadeClassifier& face_detector,
    FeatureExtractor& feature_extractor
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


std::vector<std::pair<std::string, std::vector<float>>> FileManager::loadClassifiedVectors(
    const std::string& dataset_path
){
    /*
    It retuns a vector of pairs <(class, features)> for each face in dataset.
    It receives a dataset_path with the folder where the vectors will be taken.
    
    The folder should follow this structure...
    dataset_path\
        emotion1_faces.csv
        emotion2_faces.csv
        ...
        emotionk_faces.csv
    */
    std::vector<std::pair<std::string, std::vector<float>>> classified_vectors;
    
    if (!fs::exists(dataset_path) || !fs::is_directory(dataset_path)) {
        std::cerr << "Error: Dataset path does not exist or is not a directory." << std::endl;
        return classified_vectors;
    }

    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
        
            std::string file_path = entry.path().string();
            std::string file_name = entry.path().filename().string();            
            size_t underscore_pos = file_name.find("_");
            std::string class_name = file_name.substr(0, underscore_pos);
            
            std::ifstream file(file_path); 
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file " << file_path << std::endl;
                continue; // Skip to the next file if this one fails to open
            }

            std::string line;
            while (std::getline(file, line)) {
                std::vector<float> features;
                std::stringstream ss(line);
                std::string feature_str;
                while (std::getline(ss, feature_str, ',')) {
                    try {
                        features.push_back(std::stof(feature_str));
                    } catch (const std::invalid_argument& e) {
                        std::cerr << "Warning: Invalid float value in line from file " << file_path << std::endl;
                        features.clear();
                        break;
                    }
                }
                if (!features.empty()) {
                    classified_vectors.push_back({class_name, features});
                }
            }
            file.close();
        }
    }
    
    return classified_vectors;
}