#include "classifier.h"


void saveVectorIntoCsv(
    const std::vector<float>& face_vector,
    uint8_t type_index, 
    std::ofstream& analysis_file) 
{
    /*
    This function saves a vectorized face into a CSV file.
    Parameters:
    - face_vector: The vectorized features of the face.
    - type_index: An index representing the type of face (0 for Angry, 1 for Happy, 2 for Sad).
    - analysis_file: An output file stream to write the vectorized face data.
    */

    analysis_file << type_index << ",";
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        analysis_file << face_vector[i];
        if (i < VECTOR_LENGTH - 1)
            analysis_file << ",";
    }
    analysis_file << "\n";
}


void processFolder(
    FaceClassifier& classifier,
    std::string dir_path,
    uint8_t type_index,
    std::ofstream& analysis_file
){
    /*
    This function processes a folder containing images of faces, detects faces in each image,
    vectorizes the detected faces, and writes the results to a CSV file.
    Parameters:
    - dir_path: The path to the directory containing the images.
    - type_index: An index representing the type of face (0 for Angry, 1 for Happy, 2 for Sad).
    - analysis_file: An output file stream to write the vectorized face data.
    */

    std::cout << "*Processing directory: " << dir_path << "\n";

    for (const auto& entry : fs::directory_iterator(dir_path)) {
 
        // Load image 
        std::string file_path = entry.path().string();
        cv::Mat img = cv::imread(file_path);
        if (img.empty()) {
            std::cerr << "Error al cargar la imagen: " << file_path << "\n";
            continue;
        }

        // Detect faces
        std::vector<cv::Rect> faces;
        classifier.detectFaces(img, faces);
 
        // Vectorize each detected face and write to the CSV file
        for (const auto& face : faces) {
            cv::Mat face_img = img(face).clone();
            std::vector<float> face_vector = classifier.vectorizeFace(face_img);
            saveVectorIntoCsv(face_vector, type_index, analysis_file);
        }
    }
}


void vectorizeDataset(std::string original_dataset_path){   
    /*
    This function generates a dataset of vectorized face images from the specified directories.
    It processes images from every subdirectory, which represents a face type.
    
    It vectorizes the faces and saves the results in a CSV file named "vectorized_faces.csv" 
    within the given directory.
    */

    // Open vectors file and create first row of the CSV file with headers
    std::string vectorized_dataset_path = original_dataset_path + "/vectorized_faces.csv";
    std::ofstream analysis_file;
    analysis_file.open(vectorized_dataset_path, std::ios::trunc | std::ios::binary);
    analysis_file << "Type,";
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        analysis_file << "Feature_" << i;
        if (i < VECTOR_LENGTH - 1)
            analysis_file << ",";
    }
    analysis_file << "\n";

    // Insert datasets for each face type
    uint8_t type_index = 0; 
    FaceClassifier classifier; // Create an instance of FaceClassifier
    for (const auto& entry : fs::directory_iterator(original_dataset_path)) {
        if (entry.is_directory()) {
            std::string dir_name = entry.path().filename().string();
            processFolder(classifier, entry.path().string(), type_index, analysis_file);
            type_index++;
        }
    }

    // Close the file
    analysis_file.close();
}


int main(){
    std::cout << "-----------------------------------\n";
    std::cout << "> Generating dataset...\n";
    vectorizeDataset("dataset/fer2013");
    std::cout << "Dataset generation completed.\n";
    std::cout << "-----------------------------------\n";
    return 0;
}