#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include "constants.h"

struct PCAResult {
    std::vector<std::vector<float>> basis;
    std::vector<float> mean_vector;
};

// Configuración IVF
struct IVFConfig {
    int num_clusters = 100;    // C
    int dim = PCA_DIM;         // D
    int kmeans_iters = 50;
    int nprobe = 10;
    int k = 5;
};

// Compatibilidad con código existente (lectura CSV)
struct LabeledVectors {
    std::vector<std::vector<float>> vectors;
    std::vector<int> labels;
};

#endif // TYPES_H
