#include "classifier.h"

int main(){
    FaceClassifier face_classifier;
    face_classifier.vectorizeDataset(
        "dataset/vectorized_faces.csv",
        "dataset/faces/"
    );
    return 0;
}