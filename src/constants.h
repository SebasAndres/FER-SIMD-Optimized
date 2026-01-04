#ifndef CONSTANTS_H
#define CONSTANTS_H

#define USE_ASM_IMP 1

#define IMG_SIZE 48
#define IMG_PIXELS (IMG_SIZE * IMG_SIZE)  // 2304

#define PCA_DIM 100

#define HOG_DIM 900  // HOG: 5x5 bloques x 36 features
#define HOG_BLOCK_SIZE 16
#define HOG_BLOCK_STRIDE 8
#define HOG_CELL_SIZE 8
#define HOG_NBINS 9            

#define NUM_IVF_CLUSTERS 100

inline constexpr float SUBSAMPLE_RATIO = 1.0f;

#define NUM_EMOTIONS 3
inline const std::vector<std::string> EMOTION_CATEGORIES = {
    "happy", "neutral", "sad"
};

#endif
