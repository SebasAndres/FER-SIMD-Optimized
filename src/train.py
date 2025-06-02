import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from collections import Counter

N_ESTIMATORS = 150
MAX_DEPTH = 10

def export_model(model):
    with open("forest.cpp", "w") as f:
        f.write("#include \"forest.h\"\n\n")
        for tree_idx, estimator in enumerate(model.estimators_):
            tree = estimator.tree_
            f.write(f"DecisionTreeNode forest_{tree_idx}[{len(tree.feature)}] = {{\n")
            for i in range(len(tree.feature)):
                if tree.children_left[i] == -1 and tree.children_right[i] == -1:
                    # Hoja
                    class_counts = tree.value[i].flatten()
                    predicted_class = np.argmax(class_counts)
                    f.write(f"  {{{-1}, {tree.threshold[i]}, {-1}, {-1}, {predicted_class}}}")
                else:
                    # Nodo interno
                    threshold = tree.threshold[i]
                    if np.isnan(threshold):
                        threshold = 0.0
                    f.write(f"  {{{tree.feature[i]}, {threshold}f, {tree.children_left[i]}, {tree.children_right[i]}, {-1}}}")                    
                if i < len(tree.feature) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")

            f.write("};\n\n")

        # Escribir arreglo de punteros
        f.write("DecisionTreeNode* forest[NUM_TREES] = {\n")
        for tree_idx, _ in enumerate(model.estimators_):
            f.write(f"    forest_{tree_idx},\n")
        f.write("};\n")

if __name__ == "__main__":

    # Load dataset
    dataset = pd.read_csv('dataset/vectorized_faces.csv')
    X = dataset.drop(columns=['Type'])
    Y = dataset['Type']
    print(f"Dataset shape: {X.shape}, Number of classes: {len(Y.unique())}")

    # Balance dataset
    print("[1] Balancing dataset using SMOTE...")
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(X, Y)
    print(f"> Original dataset shape: {Counter(Y)}")
    print(f"> Resampled dataset shape: {Counter(y_resampled)}")

    # Encode labels
    print("[2] Encoding labels...")
    le = LabelEncoder()
    y_balanced = le.fit_transform(y_resampled)

    # Split dataset
    print("[3] Splitting dataset into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        x_resampled,
        y_resampled,
        test_size=0.1, 
        random_state=42
    )
    print(f"> Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    # Train model
    print("[4] Training Random Forest Classifier...")
    print(f"> Number of estimators: {N_ESTIMATORS}, Max depth: {MAX_DEPTH}")   
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=1, 
        class_weight='balanced'
    )
    model.fit(x_train, y_train)

    # Evaluate model
    print("[5] Evaluating model...")
    score  = model.score(x_test, y_test)
    print(f"> Model accuracy: {score:.2f}")

    # Export model
    print("[6] Exporting model to C++ code...")
    export_model(model)