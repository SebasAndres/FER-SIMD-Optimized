#include "classifier.h"

// ============================================================================
// FaceClassifier (base)
// ============================================================================

FaceClassifier::FaceClassifier()
    : mean_vector(nullptr), mean_vector_dim(0),
      pca_basis(nullptr), pca_num_components(0), pca_vector_dim(0)
{
    loadMeanVector();
    loadPCABasis();
}

FaceClassifier::~FaceClassifier() {
    if (mean_vector) free(mean_vector);
    if (pca_basis) freeMatrix(pca_basis, pca_num_components);
}

void FaceClassifier::loadMeanVector() {
    int rows, cols;
    float* data = csv::readMatrixRaw("../data/processed/mean_vector.csv", rows, cols);
    if (!data) {
        std::cerr << "Error: No se pudo cargar el vector medio\n";
        return;
    }
    mean_vector = data;
    mean_vector_dim = cols;
}

void FaceClassifier::loadPCABasis() {
    int rows, cols;
    float* data = csv::readMatrixRaw("../data/processed/pca_basis.csv", rows, cols);
    if (!data) {
        std::cerr << "Error: No se pudo cargar la base PCA\n";
        return;
    }

    pca_num_components = rows;
    pca_vector_dim = cols;

    pca_basis = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; ++i) {
        pca_basis[i] = (float*)malloc(cols * sizeof(float));
        memcpy(pca_basis[i], data + i * cols, cols * sizeof(float));
    }
    free(data);
}

float* FaceClassifier::vectorizeFace(const cv::Mat& face_img) {
    // Resize a tamaño estándar
    cv::Mat resized_face;
    cv::resize(face_img, resized_face, cv::Size(IMG_SIZE, IMG_SIZE));

    // Conversión a blanco y negro
    cv::Mat bn_face;
    if (resized_face.channels() == 3) {
        cvtColor(resized_face, bn_face, cv::COLOR_BGR2GRAY);
    } else {
        bn_face = resized_face;
    }

    // Ecualización de histograma
    cv::equalizeHist(bn_face, bn_face);

    // Extracción de features HOG
    size_t hog_size;
    float* hog_features = extractHOG(bn_face, &hog_size);

    // Restar vector medio
    float* centered = centerVector(hog_features, mean_vector, hog_size);
    free(hog_features);

    // Proyección PCA
    float* result = projectIntoPCA(centered, pca_basis, pca_num_components, pca_vector_dim);
    free(centered);

    return result;
}

// ============================================================================
// IMPL: CentroidClassifier
// ============================================================================

CentroidClassifier::CentroidClassifier() : FaceClassifier(), centroid_dim(PCA_DIM) {
    loadCategoryCentroids();
}

CentroidClassifier::~CentroidClassifier() {
    for (auto& [label, centroid] : category_centroids) {
        if (centroid) free(centroid);
    }
}

void CentroidClassifier::loadCategoryCentroids() {
    std::string pca_faces_dir = "../data/processed/pca_faces";
    for (const auto& category : EMOTION_CATEGORIES) {
        std::string file_path = pca_faces_dir + "/" + category + "_faces.csv";

        int rows, cols;
        float* data = csv::readMatrixRaw(file_path, rows, cols);
        if (!data || rows == 0) {
            std::cerr << "Warning: No hay vectores para categoría " << category << "\n";
            continue;
        }

        float** vectors = (float**)malloc(rows * sizeof(float*));
        for (int i = 0; i < rows; ++i) {
            vectors[i] = data + i * cols;
        }

        float* centroid = calculateMeanVector(vectors, rows, cols);
        category_centroids[category] = centroid;
        centroid_dim = cols;

        free(vectors);  
        free(data);

        std::cout << "  Cargado centroide: " << category << " (" << rows << " vectores)\n";
    }
}

std::string CentroidClassifier::classifyFace(const cv::Mat& face_img) {
    float* face_vector = vectorizeFace(face_img);

    constexpr int k = 3;
    std::priority_queue<std::pair<float, std::string>> knn_queue;

    for (const auto& [label, centroid] : category_centroids) {
        float dist = euclideanDistance(face_vector, centroid, centroid_dim);
        knn_queue.push({-dist, label});
    }

    std::vector<std::string> neighbors;
    int n = std::min(k, (int)category_centroids.size());
    for (int i = 0; i < n && !knn_queue.empty(); ++i) {
        neighbors.push_back(knn_queue.top().second);
        knn_queue.pop();
    }

    std::map<std::string, int> label_count;
    for (const std::string& label : neighbors) {
        label_count[label]++;
    }

    std::string predicted = "";
    int max_count = 0;
    for (const auto& [label, count] : label_count) {
        if (count > max_count) {
            max_count = count;
            predicted = label;
        }
    }

    free(face_vector);
    return predicted.empty() ? "unknown" : predicted;
}

// ============================================================================
// IMPL: IVFClassifier (KNN real con índice IVF)
// ============================================================================

IVFClassifier::IVFClassifier() : FaceClassifier(),
    centroids(nullptr), vectors(nullptr), cluster_offsets(nullptr), labels(nullptr),
    num_clusters(0), num_vectors(0), dim(PCA_DIM), nprobe(20), k(5)
{
    loadIVFIndex();
}

IVFClassifier::~IVFClassifier() {
    if (centroids) std::free(centroids);
    if (vectors) std::free(vectors);
    if (cluster_offsets) delete[] cluster_offsets;
    if (labels) delete[] labels;
}

void IVFClassifier::loadIVFIndex() {
    std::cout << "Cargando índice IVF...\n";

    // Cargo centroides [C x D]
    int c_rows, c_cols;
    centroids = csv::readMatrixRaw("../data/processed/ivf/centroids.csv", c_rows, c_cols);
    if (!centroids) {
        std::cerr << "Error: No se pudo cargar centroids.csv\n";
        return;
    }
    num_clusters = c_rows;
    dim = c_cols;

    // Cargo vectores ordenados por cluster [N x D]
    int v_rows, v_cols;
    vectors = csv::readMatrixRaw("../data/processed/ivf/vectors.csv", v_rows, v_cols);
    if (!vectors) {
        std::cerr << "Error: No se pudo cargar vectors.csv\n";
        return;
    }
    num_vectors = v_rows;

    // Cargo cluster_offsets [C+1]
    int o_count;
    cluster_offsets = csv::readIntArray("../data/processed/ivf/cluster_offsets.csv", o_count);
    if (!cluster_offsets) {
        std::cerr << "Error: No se pudo cargar cluster_offsets.csv\n";
        return;
    }

    // Cargo labels ordenados [N]
    int l_count;
    labels = csv::readIntArray("../data/processed/ivf/labels.csv", l_count);
    if (!labels) {
        std::cerr << "Error: No se pudo cargar labels.csv\n";
        return;
    }

    std::cout << "  Índice IVF cargado:\n";
    std::cout << "    - Clusters: " << num_clusters << "\n";
    std::cout << "    - Vectores: " << num_vectors << "\n";
    std::cout << "    - Dimensiones: " << dim << "\n";
}

std::string IVFClassifier::classifyFace(const cv::Mat& face_img) {
    float* face_vector = vectorizeFace(face_img);

    // [1] Encuentro los nprobe clusters más cercanos
    std::priority_queue<
        std::pair<float, int>,
        std::vector<std::pair<float, int>>,
        std::greater<std::pair<float, int>>
    > centroids_heap;
    for (int c = 0; c < num_clusters; c++) {
        float* cluster_centroid = centroids + c * dim;
        float distance = euclideanDistance(face_vector, cluster_centroid, dim);
        centroids_heap.push({distance, c});
    }
    std::vector<int> target_clusters;
    for (int r = 0; r < nprobe && !centroids_heap.empty(); r++) {
        target_clusters.push_back(centroids_heap.top().second);
        centroids_heap.pop();
    }

    // [2] Busco solo en los vectores de los nprobe clusters 
    std::priority_queue<
        std::pair<float, int>,
        std::vector<std::pair<float, int>>,
        std::greater<std::pair<float, int>>
    > distances;
    for (int c_id : target_clusters) {
        int start = cluster_offsets[c_id];
        int end = cluster_offsets[c_id + 1];

        for (int i = start; i < end; i++) {
            int reduced_label = labels[i];
            if (reduced_label < 0) continue;
            float distance = euclideanDistance(face_vector, vectors + i * dim, dim);
            distances.push({distance, reduced_label});
        }
    }

    // [3] Extraigo los k vecinos más cercanos y 
    //  voto ponderado por distancia.
    float votes[NUM_EMOTIONS] = {0.0f};
    std::vector<std::pair<float, int>> neighbors;
    for (int j = 0; j < k && !distances.empty(); j++) {
        neighbors.push_back(distances.top());
        distances.pop();
    }
    for (int j = 0; j < (int)neighbors.size(); j++) {
        int category_label = neighbors[j].second;
        float dist = neighbors[j].first;
        float weight = 1.0f / (dist + 1e-6f);
        votes[category_label] += weight;
    }

    float max_weight = -1.0f;
    int max_voted_category = 0;
    for (int w = 0; w < NUM_EMOTIONS; w++) {
        if (votes[w] > max_weight) {
            max_weight = votes[w];
            max_voted_category = w;
        }
    }

    free(face_vector);
    return EMOTION_CATEGORIES[max_voted_category];
}