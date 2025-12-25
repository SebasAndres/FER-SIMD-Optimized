#include "classifier.h"

FaceClassifier::FaceClassifier() {
    if (!face_detector.load("face_detection/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading Haar\n";
        exit(1);
    }

    // load mean vector
    std::ifstream mean_file("data/mean_vector.csv");
    if (!mean_file.is_open()) {
        std::cout << "Could not load mean vector file\n";
    }
    else {
        std::string line;
        std::getline(mean_file, line);
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            mean_vector.push_back(std::stof(value));
        }
        mean_file.close();
    }

    // load PCA basis
    std::ifstream pca_file("data/pca_basis.csv");
    if (!pca_file.is_open()) {
        std::cout << "Could not load PCA basis\n";
    } else {
        // load matrix
        std::string line;
        while (std::getline(pca_file, line)) {
            std::vector<float> pca_vector;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) {
                pca_vector.push_back(std::stof(value));
            }
            pca_basis.push_back(pca_vector);
        }

        // calculate mean vectors (centroids) for each category
        std::string pca_faces_dir = "data/pca_faces";
        for (const auto& entry : fs::directory_iterator(pca_faces_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                std::string filename = entry.path().filename().string();
                // Extract category name from filename (e.g., "angry_faces.csv" -> "angry")
                std::string name = filename.substr(0, filename.find("_faces.csv"));

                std::vector<std::vector<float>> vectors = loadVectorsFromCsv(entry.path().string());
                std::vector<float> category_mean_vector = calculateMeanVector(vectors);
                categories_mean_vector[name] = category_mean_vector;
            }
        }
    }
}

std::vector<std::vector<float>> FaceClassifier::loadVectorsFromCsv(const std::string& file_path) {
    /*
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

FaceClassifier::~FaceClassifier() {
}


std::vector<float> FaceClassifier::vectorizeFace(const cv::Mat& face_img) {
    /*
    This function vectorizes a face image using HOG (Histogram of Oriented Gradients).
    It converts the image to grayscale, resizes it to a fixed size, and computes the HOG features.    

    Parameters: 
    - face_img: The input face image in BGR format.
    */

    // [1] Resize the image to a fixed size
    cv::Mat resized_face;
    cv::resize(face_img, resized_face, cv::Size(PROCESSED_IMG_SIZE, PROCESSED_IMG_SIZE));

    // [2] Convert the image to grayscale (if it's not already)
    cv::Mat bn_face;
    cvtColor(resized_face, bn_face, cv::COLOR_BGR2GRAY);

    // [3] Plain image into 1D vector
    std::vector<float> vector = projectInto1D(bn_face);

    // [4] Substract the means    
    std::vector<float> vector_normalized = centerVector(vector, this->mean_vector); 

    // [5] Project vector into PCA space
    std::vector<float> result = projectIntoPCA(vector, this->pca_basis);

    return result;
}


void FaceClassifier::detectFaces(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
    /*
    This function detects faces in a given frame using a Haar Cascade classifier.
    It converts the frame to grayscale and applies the face detection algorithm.

    Parameters:
    - frame: The input image frame in BN format.
    - faces: A vector to store the detected face rectangles.
    */

    face_detector.detectMultiScale(frame, faces);
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

    // k value (small odd number, e.g. 3 or 5)
    constexpr int k = 3;

    // Prepare neighbors {distance, label}
    std::priority_queue<std::pair<float, std::string>> knn_queue;

    // Preload all mean vectors per category
    // categories_mean_vector: map<label, mean_vector>

    for (const auto& [label, mean_vec] : this->categories_mean_vector) {
        float dist = euclideanDistance(face_vector, mean_vec);
        knn_queue.push({-dist, label}); // use negative distance for min-heap
    }

    // Gather top-k nearest neighbors
    std::vector<std::string> neighbors;
    int n = std::min(k, (int)this->categories_mean_vector.size());
    for (int i = 0; i < n && !knn_queue.empty(); ++i) {
        neighbors.push_back(knn_queue.top().second);
        knn_queue.pop();
    }

    // Majority vote
    std::map<std::string, int> label_count;
    for (const std::string& label : neighbors) {
        label_count[label]++;
    }

    // Find most frequent label (may break ties arbitrarily)
    std::string predicted = "";
    int max_count = 0;
    for (const auto& [label, count] : label_count) {
        if (count > max_count) {
            max_count = count;
            predicted = label;
        }
    }

    return predicted.empty() ? "unknown" : predicted;
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