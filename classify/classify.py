import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def export_model():
    with open("forest.c", "w") as f:
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
                    f.write(f"  {{{tree.feature[i]}, {tree.threshold[i]}, {tree.children_left[i]}, {tree.children_right[i]}, {-1}}}")
                if i < len(tree.feature) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("};\n\n")

    
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

    # Initialize RandomForest with reproducibility and reasonable defaults
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(x_train, y_train)
    
    score = model.score(x_test, y_test)
    print(f"Model accuracy: {score * 100:.2f}%")

    export_model()