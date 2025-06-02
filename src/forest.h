
typedef struct {
    int feature_index;   // Índice de la característica a evaluar
    float threshold;     // Umbral para comparar
    int left;            // Índice del hijo izquierdo
    int right;           // Índice del hijo derecho
    int value;           // Clase (solo en hoja)
} DecisionTreeNode;

#define NUM_TREES 150
#define MAX_NODES 1024
