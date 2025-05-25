#include "classifier.h"


FaceClassifier::FaceClassifier() {
    // Constructor
    if (!face_detector.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error al cargar el modelo Haar\n";
        exit(1);
    }

    load_centroids("centroids.dat");
}


FaceClassifier::~FaceClassifier() {
    // Destructor
}


void FaceClassifier::train() {
    std::cout << "Training the model...\n";
    std::ofstream analysis_file;
    analysis_file.open("dataset/vectorized_faces.csv", std::ios::app | std::ios::binary);
    calculate_centroid("dataset/Angry", 0, analysis_file);
    calculate_centroid("dataset/Happy", 1, analysis_file);
    calculate_centroid("dataset/Sad", 2, analysis_file);
    save_centroids("centroids.dat");
    analysis_file.close();
    std::cout << "Training the model... Done\n";
}


void FaceClassifier::calculate_centroid(std::string dir_path, uint8_t type_index, std::ofstream& analysis_file){

    size_t num_files = std::distance(fs::directory_iterator(dir_path), fs::directory_iterator{});
    std::cout << "Calculating centroid for " << dir_path << " with " << num_files << " files\n";
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        std::string file_path = entry.path().string();
        cv::Mat img = cv::imread(file_path);
        if (img.empty()) {
            std::cerr << "Error al cargar la imagen: " << file_path << "\n";
            continue;
        }
        std::vector<cv::Rect> faces;
        detect_faces(img, faces);
        for (const auto& face : faces) {
            cv::Mat face_img = img(face).clone();
            float* face_vector_ptr = vectorize_face(face_img);
            analysis_file << FACE_TYPES[type_index] << ",";
            for (int i = 0; i < VECTOR_LENGTH; ++i) {
                centroids[type_index][i] += face_vector_ptr[i] / num_files;
                analysis_file << face_vector_ptr[i];
                if (i < VECTOR_LENGTH - 1)
                    analysis_file << ",";
            }
            analysis_file << "\n";
            delete[] face_vector_ptr;
        }
    }
}


void FaceClassifier::save_centroids(const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "No se pudo abrir el archivo para guardar.\n";
        return;
    }
    for (int c = 0; c < 3; ++c) {
        file.write(reinterpret_cast<const char*>(centroids[c]), VECTOR_LENGTH * sizeof(float));
    }
    file.close();
    std::cout << "Centroides guardados en: " << filename << "\n";
}


bool FaceClassifier::load_centroids(const std::string& filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file) {
        std::cerr << "No se pudo abrir el archivo para cargar.\n";
        return false;
    }
    for (int c = 0; c < 3; ++c) {
        file.read(reinterpret_cast<char*>(centroids[c]), VECTOR_LENGTH * sizeof(float));
        if (!file) {
            std::cerr << "Error leyendo el archivo.\n";
            return false;
        }
    }
    file.close();
    std::cout << "Centroides cargados desde: " << filename << "\n";
    return true;
}


void FaceClassifier::detect_faces(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    face_detector.detectMultiScale(gray_frame, faces);
}


float* FaceClassifier::vectorize_face(const cv::Mat& face_img) {
    float* vector = new float[VECTOR_LENGTH];
    cv::Mat resized, normalized, flat;
    cv::resize(face_img, resized, cv::Size(PROCESSED_IMG_SIZE, PROCESSED_IMG_SIZE));
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
    normalized.reshape(1, 1).copyTo(flat);
    std::copy(flat.begin<float>(), flat.end<float>(), vector);
    return vector;
}


float euclidean_distance(const float* a, const float* b) {
    float sum = 0.0f;
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}


std::string FaceClassifier::run_classification(const float* face_vector) {
    float min_distance = std::numeric_limits<float>::max();
    int min_index = -1;
    for (int i = 0; i < 3; ++i) {
        float distance = euclidean_distance(face_vector, centroids[i]);
        std::cout << "Distance to " << FACE_TYPES[i] << ": " << distance << "\n";
        if (distance < min_distance) {
            min_distance = distance;
            min_index = i;
        }
    }
    return FACE_TYPES[min_index];
}

std::string FaceClassifier::classify_face(const cv::Mat& face_img) {
    float* face_vector = vectorize_face(face_img);
    std::string face_type = run_classification(face_vector);
    delete[] face_vector;
    return face_type;
}