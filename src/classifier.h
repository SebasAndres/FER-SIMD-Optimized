#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <filesystem>
#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "linalg.h"
#include "csv_utils.h"
#include "types.h"
#include "constants.h"

namespace fs = std::filesystem;

class FaceClassifier {
public:
    FaceClassifier();
    virtual ~FaceClassifier();

    void detectFaces(const cv::Mat& gray_frame, std::vector<cv::Rect>& faces);
    float* vectorizeFace(const cv::Mat& face_img);
    virtual std::string classifyFace(const cv::Mat& face_img) = 0;

protected:
    float* mean_vector;
    size_t mean_vector_dim;

    float** pca_basis;
    size_t pca_num_components;
    size_t pca_vector_dim;

    void loadMeanVector();
    void loadPCABasis();
};

// ============================================================================
class CentroidClassifier : public FaceClassifier {
public:
    CentroidClassifier();
    ~CentroidClassifier() override;

    std::string classifyFace(const cv::Mat& face_img) override;

private:
    std::map<std::string, float*> category_centroids;
    size_t centroid_dim;

    void loadCategoryCentroids();
};

// ============================================================================
class IVFClassifier : public FaceClassifier {
public:
    IVFClassifier();
    ~IVFClassifier() override;

    std::string classifyFace(const cv::Mat& face_img) override;

private:
    float* centroids;        // [C x D] centroides de clusters
    float* vectors;          // [N x D] vectores ordenados por cluster
    int* cluster_offsets;    // [C+1] offset donde empieza cada cluster
    int* labels;             // [N] labels ordenados por cluster

    int num_clusters;        // C
    int num_vectors;         // N
    int dim;                 // D
    int nprobe;              // Clusters a explorar
    int k;                   // Vecinos para KNN

    void loadIVFIndex();
};

#endif // CLASSIFIER_H
