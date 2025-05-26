#include "classifier.h"


FaceClassifier::FaceClassifier() {
    // Constructor
    if (!face_detector.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error al cargar el modelo Haar\n";
        exit(1);
    }
}


FaceClassifier::~FaceClassifier() {
    // Destructor
}


void FaceClassifier::train() {
    std::cout << "Training the model...\n";
    std::ofstream analysis_file;
    analysis_file.open("dataset/vectorized_faces.csv", std::ios::trunc | std::ios::binary);
    analysis_file << "Type,";
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        analysis_file << "Feature_" << i;
        if (i < VECTOR_LENGTH - 1)
            analysis_file << ",";
    }
    analysis_file << "\n";
    insert_dataset("dataset/Angry", 0, analysis_file);
    insert_dataset("dataset/Happy", 1, analysis_file);
    insert_dataset("dataset/Sad", 2, analysis_file);
    analysis_file.close();
    std::cout << "Training the model... Done\n";
}


std::vector<float> FaceClassifier::vectorize_face(const cv::Mat& face_img) {
    std::vector<float> vector;
    cv::Mat gray;
    cvtColor(face_img, gray, cv::COLOR_BGR2GRAY);

    // Use a fixed winSize for HOG to ensure consistent vector size
    cv::Size winSize(64, 128); 
    cv::HOGDescriptor hog(winSize, cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);

    if (gray.size() != winSize) {
        cv::resize(gray, gray, winSize);
    }
    hog.compute(gray, vector);

    return vector;
}


void FaceClassifier::insert_dataset(
    std::string dir_path,
    uint8_t type_index,
    std::ofstream& analysis_file
){

    std::cout << "*Processing directory: " << dir_path << "\n";

    size_t num_files = std::distance(fs::directory_iterator(dir_path), fs::directory_iterator{});
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
            std::vector<float> face_vector_ptr = vectorize_face(face_img);
            analysis_file << FACE_TYPES[type_index] << ",";
            for (int i = 0; i < VECTOR_LENGTH; ++i) {
                analysis_file << face_vector_ptr[i];
                if (i < VECTOR_LENGTH - 1)
                    analysis_file << ",";
            }
            analysis_file << "\n";
        }
    }
}


void FaceClassifier::detect_faces(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    face_detector.detectMultiScale(gray_frame, faces);
}


int predict_tree(int tree_idx, std::vector<float> features) {
    int node = 0;
    while (1) {
        if (forest[tree_idx][node].left == -1 &&
            forest[tree_idx][node].right == -1) {
            return forest[tree_idx][node].value;
        }
        if (features[forest[tree_idx][node].feature_index] <= forest[tree_idx][node].threshold) {
            node = forest[tree_idx][node].left;
        } else {
            node = forest[tree_idx][node].right;
        }
    }
}


std::string FaceClassifier::run_classification(std::vector<float> face_vector) {
    int votes[3] = {0}; 
    for (int t = 0; t < NUM_TREES; ++t) {
        int predicted_class = predict_tree(t, face_vector);
        votes[predicted_class]++;
    }
    int max_votes = -1;
    int predicted_class = -1;
    for (int c = 0; c < 3; ++c) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            predicted_class = c;
        }
        std::cout << FACE_TYPES[c] << ": " << votes[c] << " ";
    }
    std::cout << "\n";
    return FACE_TYPES[predicted_class];
}


std::string FaceClassifier::classify_face(const cv::Mat& face_img) {
    std::vector<float> face_vector = vectorize_face(face_img);
    std::string face_type = run_classification(face_vector);
    return face_type;
}