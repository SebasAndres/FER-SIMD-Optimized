#include "file_manager.h"
#include "face_classifier.h"
#include "linalg.h"

#define PROCESSED_IMG_SIZE 48

int main(){
    
    cv::CascadeClassifier face_detector;
    if (!face_detector.load("data/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading Haar Cascade XML file\n";
        return -1;
    }
    
    std::vector<std::vector<float>> faces_matrix;
    PCAFeatureExtractor feature_extractor("dataset/fer2013");  

    FileManager file_manager;    
    file_manager.extractFacesToCSV(
        "dataset/fer2013",
        "data/extracted_faces",
        faces_matrix,
        face_detector,
        feature_extractor
    );

    return 0;
}