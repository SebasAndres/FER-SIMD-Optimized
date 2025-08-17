#include "feature_extractor.h"

#define PROCESSED_IMG_SIZE 48

SimpleFeatureExtractor::SimpleFeatureExtractor(std::string dataset_path)
    : FeatureExtractor(dataset_path) 
{

}

std::vector<float> SimpleFeatureExtractor::extractFeatures(const cv::Mat& image) {

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = image.clone();
    }
    gray.convertTo(gray, CV_8UC1); // Asegura el tipo correcto


    cv::Mat resized_face;
    cv::resize(
        gray, 
        resized_face, 
        cv::Size(PROCESSED_IMG_SIZE, PROCESSED_IMG_SIZE)
    );
    
    std::vector<float> face_vector = projectInto1D(resized_face);
    return face_vector;
}