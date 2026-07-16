import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_data():
    train_df = pd.read_csv("data/Training.csv")
    test_df = pd.read_csv("data/Testing.csv")

    # Drop the empty unnamed column if it exists
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

    return train_df, test_df

def prepare_features(train_df, test_df):
    X_train = train_df.drop("prognosis", axis=1)
    y_train = train_df["prognosis"]

    X_test = test_df.drop("prognosis", axis=1)
    y_test = test_df["prognosis"]

    # Encode disease names into numbers for the model
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    return X_train, y_train_encoded, X_test, y_test_encoded, le

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy

if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test, le = prepare_features(train_df, test_df)

    print("Training Naive Bayes model...")
    nb_model = train_naive_bayes(X_train, y_train)
    nb_accuracy = evaluate_model(nb_model, X_test, y_test, le, "Naive Bayes (Main Model)")

    print("\nTraining Decision Tree model (baseline)...")
    dt_model = train_decision_tree(X_train, y_train)
    dt_accuracy = evaluate_model(dt_model, X_test, y_test, le, "Decision Tree (Baseline)")

    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Naive Bayes Accuracy:   {nb_accuracy * 100:.2f}%")
    print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")

    nb_predictions = nb_model.predict(X_test)

    # Confusion matrix (Naive Bayes)
    cm = confusion_matrix(y_test, nb_predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Naive Bayes")
    plt.xlabel("Predicted Disease (encoded)")
    plt.ylabel("Actual Disease (encoded)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("docs/confusion_matrix_nb.png", dpi=150)
    plt.close()

    # Accuracy comparison bar chart
    plt.figure(figsize=(6, 5))
    models = ["Naive Bayes\n(Main Model)", "Decision Tree\n(Baseline)"]
    accuracies = [nb_accuracy * 100, dt_accuracy * 100]
    colors = ["#2196F3", "#FF9800"]
    plt.bar(models, accuracies, color=colors)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison: Naive Bayes vs Decision Tree")
    plt.ylim(0, 105)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("docs/model_comparison.png", dpi=150)
    plt.close()

    print("\nVisuals saved to docs/ folder: confusion_matrix_nb.png, model_comparison.png")