#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

#include "constants.h"
#include "types.h"

namespace fs = std::filesystem;

namespace csv {

std::vector<std::vector<float>> readMatrix(const std::string& file_path);

std::vector<float> readVector(const std::string& file_path);

LabeledVectors readLabeledVectors(const std::string& directory_path,
                                  const std::vector<std::string>& label_names);

void writeMatrix(const std::vector<std::vector<float>>& matrix,
                 const std::string& file_path);

void writeVector(const std::vector<float>& vec,
                 const std::string& file_path);

void writeIntVector(const std::vector<int>& vec,
                    const std::string& file_path);

// Escribir desde punteros (memoria contigua)
void writeMatrixRaw(const float* data, int rows, int cols,
                    const std::string& file_path);

void writeMatrixRaw2D(float** data, size_t rows, size_t cols,
                      const std::string& file_path);

void writeVectorRaw(const float* data, size_t size,
                    const std::string& file_path);

void writeIntArray(const int* data, int count,
                   const std::string& file_path);

// Leer a punteros (memoria contigua)
float* readMatrixRaw(const std::string& file_path, int& rows, int& cols);

int* readIntArray(const std::string& file_path, int& count);

void ensureDirectoryExists(const std::string& dir_path);

std::vector<std::string> getCategoryNames(const std::string& directory_path);

}  // namespace csv

#endif  // CSV_UTILS_H
