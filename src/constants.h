#ifndef CONSTANTS_H
#define CONSTANTS_H

// Switch entre uso funciones assembly / C
#define USE_ASM_IMP 1

// Switch entre IVFClassifier / CentroidClassifier
#define USE_IVF 1 // Es el que mejor funciona

// Orden y emotion_categories usadas
#define NUM_EMOTIONS 3
inline const std::vector<std::string> EMOTION_CATEGORIES = {
    "happy", "neutral", "sad"
};

// Dimensión de V_d
// IMPORTANTE: Debe ser múltiplo de 4 para las optimizaciones de SIMD
#define PCA_DIM 100

// Cantidad total de clusters
#define NUM_IVF_CLUSTERS 100

// Dimensión uniforme de las imagenes (resize)
#define IMG_SIZE 48
#define IMG_PIXELS (IMG_SIZE * IMG_SIZE) 

// HOG Configs
#define HOG_BLOCK_SIZE 16
#define HOG_BLOCK_STRIDE 8
#define HOG_CELL_SIZE 8
#define HOG_NBINS 9            

// IMPORTANTE: Debe ser múltiplo de 4 para las optimizaciones de SIMD
#define HOG_DIM 900  // HOG: 5x5 bloques x 36 features

// Es el mejor subsample ratio para IVF que testee
inline constexpr float SUBSAMPLE_RATIO = 1.0f;

// Definiciones de la window en la aplicación
#define FONT_LABELS cv::FONT_HERSHEY_SIMPLEX
#define FONT_SCALE 0.8
#define FONT_THICKNESS 2
#define FONT_COLOR cv::Scalar(255, 0, 0)
#define RECTANGLE_COLOR cv::Scalar(255, 0, 0)

#endif
