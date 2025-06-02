#include "classifier.h"


FaceClassifier::FaceClassifier() {
    // TODO: Deprecar
    if (!face_detector.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error al cargar el modelo Haar\n";
        exit(1);
    }

    loadFaceVectors();    
}


FaceClassifier::~FaceClassifier() {
}

void FaceClassifier::loadFaceVectors() {
    /*
    This function loads the face vectors from the dataset into the face_vectors vector.
    It initializes the face_vectors vector with the data from the dataset.
    */
   
    std::ifstream file("dataset/vectorized_faces.csv");
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo de dataset\n";
        exit(1);
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        if (line_number == 0) { // Skip header
            line_number++;
            continue;
        }        
        std::stringstream ss(line);
        int type;
        std::vector<float> vector(VECTOR_LENGTH);
        
        // Read type
        ss >> type;

        // Read vector values
        for (int i = 0; i < VECTOR_LENGTH; ++i) {
            std::string value;
            std::getline(ss, value, ',');
            vector[i] = std::stof(value);
        }

        // Store the case
        FaceNode face_case;
        face_case.type = type;
        face_case.vector = vector;
        FACE_NODES.push_back(face_case);
        line_number++;
    }
}


void FaceClassifier::vectorizeDataset(
    std::string vectorized_dataset_path,
    std::string original_dataset_path
) {   
    /*
    This function generates a dataset of vectorized face images from the specified directories.
    It processes images from three folders: Angry, Happy, and Sad, vectorizes the faces using HOG features,
    and saves the results in a CSV file named "vectorized_faces.csv" in the "dataset" directory.
    
    The CSV file contains the type of face (Angry, Happy, Sad) and the corresponding feature vector.
    */

    std::cout << "Generating dataset...\n";

    // Open vectors file and create first row of the CSV file with headers
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
    processFolder(original_dataset_path+"Angry", 0, analysis_file);
    processFolder(original_dataset_path+"Happy", 1, analysis_file);
    processFolder(original_dataset_path+"Sad", 2, analysis_file);

    // Close the file
    analysis_file.close();

    std::cout << "Dataset generation completed.\n";
}


std::vector<float> FaceClassifier::vectorizeFace(const cv::Mat& face_img) {
    /*
    This function vectorizes a face image using HOG (Histogram of Oriented Gradients).
    It converts the image to grayscale, resizes it to a fixed size, and computes the HOG features.    

    Parameters: 
    - face_img: The input face image in BGR format.
    */

    std::vector<float> vector;
    cv::Mat gray;
    cvtColor(face_img, gray, cv::COLOR_BGR2GRAY);

    // Use a fixed winSize for HOG to ensure consistent vector size
    cv::Size winSize(64, 128); 
    cv::HOGDescriptor hog(winSize, cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);

    if (gray.size() != winSize) 
        cv::resize(gray, gray, winSize);
    
    hog.compute(gray, vector);

    return vector;
}


void FaceClassifier::processFolder(
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
        detectFaces(img, faces);
 
        // Vectorize each detected face and write to the CSV file
        for (const auto& face : faces) {
            cv::Mat face_img = img(face).clone();
            std::vector<float> face_vector = vectorizeFace(face_img);
            
            // saveFace(face_img, type_index);
            saveVectorIntoCsv(face_vector, type_index, analysis_file);
        }
    }
}


void FaceClassifier::saveFace(const cv::Mat& face_img, uint8_t type_index) {
    /*
    This function saves a detected face image to a file.
    Parameters:
    - face_img: The detected face image in BGR format.
    - type_index: An index representing the type of face (0 for Angry, 1 for Happy, 2 for Sad).
    */
    std::string output_folder = "dataset/faces/"+FACE_TYPES[type_index]+"/";
    static int face_id = 0; // Unique ID for each face
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    std::string output_file = output_folder + "face_" + std::to_string(face_id++) + "_.jpg";
    cv::imwrite(output_file, face_img);
}



void FaceClassifier::saveVectorIntoCsv(
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



void FaceClassifier::detectFaces(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
    /*
    This function detects faces in a given frame using a Haar Cascade classifier.
    It converts the frame to grayscale and applies the face detection algorithm.

    Parameters:
    - frame: The input image frame in BGR format.
    - faces: A vector to store the detected face rectangles.
    */
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    face_detector.detectMultiScale(gray_frame, faces);
}


float FaceClassifier::computeDistance(
    const std::vector<float>& face_vector1, 
    const std::vector<float>& face_vector2
) {
    /*
    This function computes the Euclidean distance between two face vectors.
    Parameters:
    - face_vector1: The first face vector.
    - face_vector2: The second face vector.
    Returns:
    - The Euclidean distance between the two vectors.
    */
    
    float distance = 0.0f;
    for (size_t i = 0; i < VECTOR_LENGTH; ++i) {
        float diff = face_vector1[i] - face_vector2[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}


std::string FaceClassifier::runClassification(std::vector<float> face_vector) {
    /*
    This function runs the classification of a face vector using the KNN algorithm.
    This could run parallel to speed up the classification process using SIMD.

    Parameters:
    - face_vector: A vector of features extracted from the face image.
    Returns:
    - The predicted face type as a string.
    */

    struct Compare2nd {
        bool operator()(const std::pair<float, FaceNode>& a, const std::pair<float, FaceNode>& b) {
            return a.first > b.first; // Min-heap based on distance
        }
    };

    uint16_t face_type_counter[3] = {0,0,0};
    std::priority_queue<std::pair<float, FaceNode>, std::vector<std::pair<float, FaceNode>>, Compare2nd> minHeapNodes;

    // Build a min-heap to store the distances
    // This could be easily parallelized using SIMD
    for (int i = 0; i < NUM_FACE_NODES; ++i) {
        float distance = this->computeDistance(face_vector, FACE_NODES[i].vector);
        minHeapNodes.push({distance, FACE_NODES[i]});
    }

    // Get top K neighbors and its face-types
    // Then sum the face-types to determine the final classification
    for (int j=0; j < K; ++j) {        
        // Extract minimum distance node
        FaceNode node = minHeapNodes.top().second;
        minHeapNodes.pop();

        // Increment the counter for the face type
        face_type_counter[node.type]++;
    }

    // Get index of the maximum face type counter
    int max_index = 0;
    for (int i = 1; i < 3; ++i) {
        if (face_type_counter[i] > face_type_counter[max_index]) {
            max_index = i;
        }
    }

    return FACE_TYPES[max_index];
}


std::string FaceClassifier::classifyFace(const cv::Mat& face_img) {
    /*
    This function classifies a face image by first vectorizing it and then running the classification.
    Parameters:
    - face_img: The input face image in BGR format.
    Returns:
    - The predicted face type as a string.
    */

    std::vector<float> face_vector = vectorizeFace(face_img);
    std::string face_type = runClassification(face_vector);
    return face_type;
}