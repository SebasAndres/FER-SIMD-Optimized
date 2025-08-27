#include "classifier.h"
#include "file_manager.h"
#include <filesystem>
namespace fs = std::filesystem;

KNNClassifier::KNNClassifier(const std::string& dataset_path){
    this->points = FileManager::loadClassifiedVectors(dataset_path);
}


std::string KNNClassifier::classify(const std::vector<float>& features){
    return std::to_string(this->points.size());
}