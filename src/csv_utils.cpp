#include "csv_utils.h"
#include <algorithm>

namespace csv {

std::vector<std::vector<float>> readMatrix(const std::string& file_path) {
    std::vector<std::vector<float>> matrix;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "[csv::readMatrix] Error: no se pudo abrir " << file_path << "\n";
        return matrix;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        matrix.push_back(row);
    }

    file.close();
    return matrix;
}


std::vector<float> readVector(const std::string& file_path) {
    std::vector<float> vec;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "[csv::readVector] Error: no se pudo abrir " << file_path << "\n";
        return vec;
    }

    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            vec.push_back(std::stof(value));
        }
    }

    file.close();
    return vec;
}


LabeledVectors readLabeledVectors(
    const std::string& directory_path,
    const std::vector<std::string>& label_names
) {
    LabeledVectors result;
    for (size_t label_id = 0; label_id < label_names.size(); ++label_id) {
        std::string file_path = directory_path + "/" + label_names[label_id] + "_faces.csv";
        if (!fs::exists(file_path)) {
            std::cerr << "[csv::readLabeledVectors] Archivo no encontrado: " << file_path << "\n";
            continue;
        }
        std::vector<std::vector<float>> category_vectors = readMatrix(file_path);
        for (const auto& vec : category_vectors) {
            result.vectors.push_back(vec);
            result.labels.push_back(static_cast<int>(label_id));
        }
    }
    return result;
}

void writeMatrix(const std::vector<std::vector<float>>& matrix,
                 const std::string& file_path) {
    std::ofstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "[csv::writeMatrix] Error: no se pudo crear " << file_path << "\n";
        return;
    }

    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}


void writeVector(const std::vector<float>& vec, const std::string& file_path) {
    std::ofstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "[csv::writeVector] Error: no se pudo crear " << file_path << "\n";
        return;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        file << vec[i];
        if (i < vec.size() - 1) {
            file << ",";
        }
    }

    file.close();
}


void writeIntVector(const std::vector<int>& vec, const std::string& file_path) {
    std::ofstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "[csv::writeIntVector] Error: no se pudo crear " << file_path << "\n";
        return;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        file << vec[i];
        if (i < vec.size() - 1) {
            file << ",";
        }
    }

    file.close();
}

void writeMatrixRaw(const float* data, int rows, int cols,
                    const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "[csv::writeMatrixRaw] Error: no se pudo crear " << file_path << "\n";
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << data[i * cols + j];
            if (j < cols - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}


void writeMatrixRaw2D(float** data, size_t rows, size_t cols,
                      const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "[csv::writeMatrixRaw2D] Error: no se pudo crear " << file_path << "\n";
        return;
    }

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            file << data[i][j];
            if (j < cols - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}


void writeVectorRaw(const float* data, size_t size,
                    const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "[csv::writeVectorRaw] Error: no se pudo crear " << file_path << "\n";
        return;
    }

    for (size_t i = 0; i < size; i++) {
        file << data[i];
        if (i < size - 1) file << ",";
    }
    file.close();
}


void writeIntArray(const int* data, int count, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "[csv::writeIntArray] Error: no se pudo crear " << file_path << "\n";
        return;
    }

    for (int i = 0; i < count; i++) {
        file << data[i];
        if (i < count - 1) file << ",";
    }
    file.close();
}


float* readMatrixRaw(const std::string& file_path, int& rows, int& cols) {
    std::vector<std::vector<float>> matrix = readMatrix(file_path);
    if (matrix.empty()) {
        rows = cols = 0;
        return nullptr;
    }
    rows = matrix.size();
    cols = matrix[0].size();
    float* data = (float*)malloc(sizeof(float) * rows * cols);
    for (int i = 0; i < rows; i++) {
        std::memcpy(data + i * cols, matrix[i].data(), cols * sizeof(float));
    }
    return data;
}


int* readIntArray(const std::string& file_path, int& count) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "[csv::readIntArray] Error: no se pudo abrir " << file_path << "\n";
        count = 0;
        return nullptr;
    }

    std::vector<int> vec;
    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            vec.push_back(std::stoi(value));
        }
    }
    file.close();

    count = vec.size();
    int* data = new int[count];
    std::memcpy(data, vec.data(), count * sizeof(int));

    return data;
}


void ensureDirectoryExists(const std::string& dir_path) {
    if (!fs::exists(dir_path)) {
        fs::create_directories(dir_path);
    }
}


std::vector<std::string> getCategoryNames(const std::string& directory_path) {
    std::vector<std::string> names;

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::string filename = entry.path().filename().string();
            // "angry_faces.csv" -> "angry"
            size_t pos = filename.find("_faces.csv");
            if (pos != std::string::npos) {
                names.push_back(filename.substr(0, pos));
            }
        }
    }

    std::sort(names.begin(), names.end());
    return names;
}

}