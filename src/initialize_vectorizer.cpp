/* 
Este script procesa el dataset para generar:
1. Vector medio (data/processed/mean_vector.csv)
2. Base PCA (data/processed/pca_basis.csv)
3. Vectores PCA por categoría (data/processed/pca_faces/*.csv)
4. Un índice IVF para búsqueda eficiente:
    - data/processed/ivf/centroids.csv 
    - data/processed/ivf/cluster_offsets.csv
    - data/processed/ivf/labels.csv 
    - data/processed/ivf/vectors.csv  
*/

#include <random>
#include <numeric>
#include <algorithm>

#include "classifier.h"
#include "linalg.h"
#include "csv_utils.h"
#include "types.h"
#include "constants.h"

std::vector<float> getFaceHOGVector(cv::Mat face_img) {
    cv::Mat resized_face;
    cv::resize(
        face_img,
        resized_face,
        cv::Size(IMG_SIZE, IMG_SIZE)
    );
    cv::Mat equalized;
    cv::equalizeHist(resized_face, equalized);
    size_t hog_size;
    float* hog_ptr = extractHOG(equalized, &hog_size);
    std::vector<float> result(hog_ptr, hog_ptr + hog_size);
    free(hog_ptr);
    return result;
}


void extractFacesFromDirectory(
    const std::string& dir_path,
    std::function<std::vector<float>(cv::Mat)> vectorizer,
    std::vector<std::vector<float>>& output_vectors
) {
    /* Dado un directorio con imagenes, las recorre y detecta caras.
    Por cada cara detectada aplica la vectorización y la acumula en 
    output_vectors.
    */
    cv::CascadeClassifier face_detector;
    if (!face_detector.load("face_detection/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading Haar\n";
        exit(1);
    }

    std::cout << "  Procesando: " << dir_path << "\n";
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        std::string file_path = entry.path().string();
        cv::Mat img = cv::imread(file_path);
        if (img.empty()) continue;

        cv::Mat gray_img;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
        } else {
            gray_img = img.clone();
        }

        std::vector<cv::Rect> faces;
        face_detector.detectMultiScale(gray_img, faces);

        for (const auto& face : faces) {
            cv::Rect safe_face = face & cv::Rect(0, 0, gray_img.cols, gray_img.rows);
            if (safe_face.area() == 0) continue;

            cv::Mat face_img = gray_img(safe_face).clone();
            std::vector<float> vec = vectorizer(face_img);

            if (!vec.empty()) {
                output_vectors.push_back(vec);
            }
        }
    }
}

void collectAllFaces1D(
    const std::string& dataset_path,
    std::vector<std::vector<float>>& all_faces
) {
    /* Por cada tipo de cara en el dataset, obtener rostros y 
    vectorizarlos con HOG, devolviendolos de forma unificada en
    el parámetro inout all_faces y guardardolos en una carpeta por
    categoría.
    */

    std::cout << "\n[Paso 1] Recolectando caras del dataset...\n";
    csv::ensureDirectoryExists("../data/processed/faces");

    for (const auto& category : EMOTION_CATEGORIES) {
        std::string category_path = dataset_path + "/" + category;
        std::vector<std::vector<float>> category_faces;
        extractFacesFromDirectory(
            category_path, 
            getFaceHOGVector, 
            category_faces
        );
        std::string output_path = "../data/processed/faces/" + category + "_faces.csv";
        csv::writeMatrix(category_faces, output_path);

        // Agrego al conjunto total
        for (const auto& face : category_faces) {
            all_faces.push_back(face);
        }
        std::cout << "    " << category << ": " << category_faces.size() << " caras\n";
    }

    std::cout << "  Total: " << all_faces.size() << " caras\n";
}

PCAResult computeAndSavePCA(
    std::vector<std::vector<float>>& faces,
    int num_components
) {
    std::cout << "\n[Paso 2] Calculando PCA...\n";

    size_t num_vectors = faces.size();
    size_t vector_dim = faces[0].size();

    // Pasaje a float**
    float** faces_ptr = (float**)malloc(num_vectors * sizeof(float*));
    for (size_t i = 0; i < num_vectors; ++i) {
        faces_ptr[i] = faces[i].data();
    }

    // [1] Vector medio
    std::cout << "  Calculando vector medio...\n";
    float* mean_vec = calculateMeanVector(faces_ptr, num_vectors, vector_dim);
    csv::writeVectorRaw(mean_vec, vector_dim, "../data/processed/mean_vector.csv");

    // [2] Centrado de vectores
    std::cout << "  Centrando vectores...\n";
    float** centered = centerVectors(faces_ptr, mean_vec, num_vectors, vector_dim);

    // [3] Matriz de covarianza
    std::cout << "  Calculando matriz de covarianza...\n";
    cv::Mat cov_matrix = calculateCovarianceMatrix(centered, num_vectors, vector_dim);

    // [4] Base PCA
    std::cout << "  Calculando " << num_components << " componentes PCA...\n";
    size_t pca_vector_dim;
    float** pca_basis = calculatePCABasis(cov_matrix, num_components, &pca_vector_dim);
    csv::writeMatrixRaw2D(pca_basis, num_components, pca_vector_dim, "../data/processed/pca_basis.csv");
    std::cout << "  PCA guardado en data/processed/pca_basis.csv\n";

    // Convierto resultados a std::vector para PCAResult
    std::vector<float> mean_vec_result(mean_vec, mean_vec + vector_dim);
    std::vector<std::vector<float>> pca_basis_result(num_components);
    for (int i = 0; i < num_components; ++i) {
        pca_basis_result[i] = std::vector<float>(pca_basis[i], pca_basis[i] + pca_vector_dim);
    }

    free(faces_ptr);
    free(mean_vec);
    freeMatrix(centered, num_vectors);
    freeMatrix(pca_basis, num_components);

    return {pca_basis_result, mean_vec_result};
}


void projectFacesToPCA(
    const std::string& dataset_path,
    const std::vector<std::vector<float>>& pca_basis,
    const std::vector<float>& mean_vector
) {
    std::cout << "\n[Paso 3] Proyectando caras a espacio PCA...\n";
    csv::ensureDirectoryExists("../data/processed/pca_faces");

    size_t num_components = pca_basis.size();
    size_t pca_vector_dim = pca_basis[0].size();

    // Convierto pca_basis a float**
    float** pca_basis_ptr = (float**)malloc(num_components * sizeof(float*));
    for (size_t i = 0; i < num_components; ++i) {
        pca_basis_ptr[i] = (float*)pca_basis[i].data();
    }

    auto pca_vectorizer = [&](cv::Mat face) -> std::vector<float> {
        cv::Mat equalized;
        cv::equalizeHist(face, equalized);
        size_t hog_size;
        float* hog_ptr = extractHOG(equalized, &hog_size);
        float* centered = centerVector(hog_ptr, mean_vector.data(), hog_size);
        free(hog_ptr);
        float* projected = projectIntoPCA(centered, pca_basis_ptr, num_components, pca_vector_dim);
        free(centered);
        std::vector<float> result(projected, projected + num_components);
        free(projected);
        return result;
    };

    for (const auto& category : EMOTION_CATEGORIES) {
        std::string category_path = dataset_path + "/" + category;
        if (!fs::exists(category_path)) continue;
        std::vector<std::vector<float>> pca_faces;

        extractFacesFromDirectory(category_path, pca_vectorizer, pca_faces);

        std::string output_path = "../data/processed/pca_faces/" + category + "_faces.csv";
        csv::writeMatrix(pca_faces, output_path);
        std::cout << "    " << category << ": " << pca_faces.size() << " vectores PCA\n";
    }

    free(pca_basis_ptr);
}

void trainKMeans(
    float* vectors,          
    int num_vectors,        
    float* centroids,       
    int* assignments,       
    const IVFConfig& config
) {
    /* Calculo los `config.num_clusterss` centroides a partir de todos
    los vectores, aplicando iteraciones de KMeans. */

    int D = config.dim;
    int C = config.num_clusters;

    // [1] Centroides random (random sampling)
    std::vector<int> indices(num_vectors);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int c = 0; c < C; c++) {
        int idx = indices[c];
        std::memcpy(centroids + c * D, vectors + idx * D, D * sizeof(float));
    }

    // [2] Iteraciones Kmeans
    int* num_vectors_per_centroid = (int*)std::malloc(C * sizeof(int));
    for (int epoch = 0; epoch < config.kmeans_iters; epoch++) {

        // Reseteo centroides y contadores
        std::memset(centroids, 0, C * D * sizeof(float));
        std::memset(num_vectors_per_centroid, 0, C * sizeof(int));

        // Asigno cada vector al centroide mas cercano
        for (int i = 0; i < num_vectors; i++) {
            float* vector = vectors + i * D;
            float min_distance = euclideanDistance(vector, centroids, D);
            int closest_centroid_index = 0;
            for (int c = 1; c < C; c++) {
                float* centroid = centroids + c * D;
                float distance = euclideanDistance(vector, centroid, D);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_centroid_index = c;
                }
            }
            assignments[i] = closest_centroid_index;
        }

        // Acumulo vectores en sus centroides
        for (int i = 0; i < num_vectors; i++) {
            int c = assignments[i];
            float* vector = vectors + i * D;
            float* centroid = centroids + c * D;
            for (int d = 0; d < D; d++) 
                centroid[d] += vector[d];
            num_vectors_per_centroid[c]++;
        }

        // Divido por cantidad en cada centroide 
        // para obtener nuevos centroides
        for (int c = 0; c < C; c++) {
            if (num_vectors_per_centroid[c] > 0) {
                float* centroid = centroids + c * D;
                for (int d = 0; d < D; d++)
                    centroid[d] /= num_vectors_per_centroid[c];
            }
        }
    }

    std::free(num_vectors_per_centroid);
}

LabeledVectors subsampleHappy(const LabeledVectors& data, float ratio) {
    // Submuestreo de la clase mayoritaria (happy) para balancear el dataset
    std::cout << "  Aplicando submuestreo de happy (ratio " << ratio << ":1)...\n";
    std::vector<int> class_counts(NUM_EMOTIONS, 0);
    for (int label : data.labels)
        class_counts[label]++;
    std::cout << "    Distribución original:\n";
    for (int i = 0; i < NUM_EMOTIONS; i++)
        std::cout << "      " << EMOTION_CATEGORIES[i] << ": " << class_counts[i] << "\n";
    int n_sad = class_counts[2];
    int n_happy_target = static_cast<int>(n_sad * ratio);
    n_happy_target = std::min(n_happy_target, class_counts[0]);  // no más que el original
    std::vector<std::vector<int>> indices_by_class(NUM_EMOTIONS);
    for (size_t i = 0; i < data.labels.size(); i++)
        indices_by_class[data.labels[i]].push_back(i);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices_by_class[0].begin(), indices_by_class[0].end(), gen);
    indices_by_class[0].resize(n_happy_target);
    std::vector<int> selected_indices;
    for (int c = 0; c < NUM_EMOTIONS; c++) {
        for (int idx : indices_by_class[c])
            selected_indices.push_back(idx);
    }
    std::shuffle(selected_indices.begin(), selected_indices.end(), gen);
    LabeledVectors result;
    result.vectors.reserve(selected_indices.size());
    result.labels.reserve(selected_indices.size());
    for (int idx : selected_indices) {
        result.vectors.push_back(data.vectors[idx]);
        result.labels.push_back(data.labels[idx]);
    }
    std::vector<int> new_counts(NUM_EMOTIONS, 0);
    for (int label : result.labels)
        new_counts[label]++;
    std::cout << "    Distribución balanceada:\n";
    for (int i = 0; i < NUM_EMOTIONS; i++)
        std::cout << "      " << EMOTION_CATEGORIES[i] << ": " << new_counts[i] << "\n";
    std::cout << "    Total: " << result.vectors.size() << " vectores\n";
    return result;
}

void buildIVFIndex(const IVFConfig& config) {
    std::cout << "\n[Paso 4] Construyendo índice IVF...\n";

    // Cargo vectores PCA con labels
    std::cout << "  Cargando vectores PCA...\n";
    LabeledVectors raw_data = csv::readLabeledVectors("../data/processed/pca_faces", EMOTION_CATEGORIES);

    // Submuestreo de happy para balancear clases
    LabeledVectors balanced_data = subsampleHappy(raw_data, SUBSAMPLE_RATIO);
    
    // Convierto los vectores a memoria contigua
    std::cout << "  Convirtiendo a memoria contigua...\n";
    int N = balanced_data.vectors.size();
    int D = config.dim;
    int C = config.num_clusters;
    float* vectors = (float*) malloc(sizeof(float) * N * D);
    int* labels = new int[N];
    for (int i = 0; i < N; ++i) {
        std::memcpy(vectors + i * D, balanced_data.vectors[i].data(), D * sizeof(float));
        labels[i] = balanced_data.labels[i];
    }
    std::cout << "    Total vectores: " << N << "\n";
    std::cout << "    Dimensiones: " << D << "\n";

    // Calculo los centroides
    std::cout << "  Ejecutando K-means...\n";
    float* centroids = (float*) malloc(sizeof(float) * C * D);
    int* assignments = new int[N]();
    trainKMeans(vectors, N, centroids, assignments, config);

    // Reordeno vectores por cluster (listas invertidas)
    std::cout << "  Construyendo listas invertidas...\n";
    int* cluster_sizes = new int[C]();
    for (int i = 0; i < N; i++) {
        cluster_sizes[assignments[i]]++;
    }
    // cluster_offset[c] = "índice en vectors del primer elemento del cluster c"
    int* cluster_offsets = new int[C + 1];
    cluster_offsets[0] = 0;
    for (int c = 0; c < C; c++)
        cluster_offsets[c + 1] = cluster_offsets[c] + cluster_sizes[c];

    // reordeno arrays
    float* vectors_sorted = (float*) malloc(sizeof(float) * N * D);
    int* labels_sorted = new int[N];
    int* write_pos = new int[C]; // dónde escribir el siguiente elemento de cluster c
    std::memcpy(write_pos, cluster_offsets, C * sizeof(int));
    for (int i = 0; i < N; i++) {
        int c = assignments[i];
        int dest = write_pos[c];
        std::memcpy(vectors_sorted + dest * D, vectors + i * D, D * sizeof(float));
        labels_sorted[dest] = labels[i];
        write_pos[c]++;
    }
    std::cout << "    Listas invertidas construidas.\n";

    // Guardo índice IVF
    std::cout << "  Guardando índice IVF...\n";
    csv::ensureDirectoryExists("../data/processed/ivf");
    csv::writeMatrixRaw(centroids, C, D, "../data/processed/ivf/centroids.csv");
    std::cout << "    - data/processed/ivf/centroids.csv (" << C << " x " << D << ")\n";
    csv::writeMatrixRaw(vectors_sorted, N, D, "../data/processed/ivf/vectors.csv");
    std::cout << "    - data/processed/ivf/vectors.csv (" << N << " x " << D << ")\n";
    csv::writeIntArray(cluster_offsets, C + 1, "../data/processed/ivf/cluster_offsets.csv");
    std::cout << "    - data/processed/ivf/cluster_offsets.csv (" << C + 1 << " elementos)\n";
    csv::writeIntArray(labels_sorted, N, "../data/processed/ivf/labels.csv");
    std::cout << "    - data/processed/ivf/labels.csv (" << N << " elementos)\n";

    std::free(vectors);
    std::free(vectors_sorted);
    std::free(centroids);
    delete[] labels;
    delete[] labels_sorted;
    delete[] assignments;
    delete[] cluster_sizes;
    delete[] cluster_offsets;
    delete[] write_pos;

    std::cout << "  Índice IVF guardado correctamente.\n";
}

int main() {
    std::string dataset_path = "../data/dataset/fer2013";

    // [1] Obtengo todas las caras del dataset
    std::vector<std::vector<float>> all_faces;
    collectAllFaces1D(dataset_path, all_faces);

    // [2] Calculo PCA sobre todos los vectores
    int num_pca_components = PCA_DIM;
    PCAResult pca_result = computeAndSavePCA(all_faces, num_pca_components);

    // [3] Proyecto todos los vectores sobre la base PCA calculada
    projectFacesToPCA(dataset_path, pca_result.basis, pca_result.mean_vector);

    // [4] Armo y guardo datos para el IVFClassifier
    IVFConfig ivf_config;
    ivf_config.num_clusters = NUM_IVF_CLUSTERS;
    ivf_config.dim = PCA_DIM;
    buildIVFIndex(ivf_config);

    std::cout << "Archivos generados\n";
    return 0;
}
