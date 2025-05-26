import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from collections import Counter

def export_model():
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

    # Separate features and target
    X = dataset.drop(columns=['Type'])
    y = dataset['Type']


    # Split into train and test sets with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(Counter(y_train))

    # Initialize RandomForest with reproducibility and reasonable defaults
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(x_train, y_train)
    
    score = model.score(x_test, y_test)
    print(f"Model accuracy: {score * 100:.2f}%")

    export_model()