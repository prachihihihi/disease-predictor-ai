#Prachi Silwal
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_data():
    train_df = pd.read_csv("data/Training.csv")
    test_df = pd.read_csv("data/Testing.csv")

    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

    return train_df, test_df

def prepare_features(train_df, test_df):
    X_train = train_df.drop("prognosis", axis=1)
    y_train = train_df["prognosis"]

    X_test = test_df.drop("prognosis", axis=1)
    y_test = test_df["prognosis"]

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    return X_train, y_train_encoded, X_test, y_test_encoded, le

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_bernoulli_nb(X_train, y_train):
    model = BernoulliNB()
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

    print("Training Gaussian Naive Bayes model...")
    nb_model = train_naive_bayes(X_train, y_train)
    nb_accuracy = evaluate_model(nb_model, X_test, y_test, le, "Gaussian Naive Bayes (Initial Model)")

    print("\nTraining Bernoulli Naive Bayes model (better suited for binary features)...")
    bnb_model = train_bernoulli_nb(X_train, y_train)
    bnb_accuracy = evaluate_model(bnb_model, X_test, y_test, le, "Bernoulli Naive Bayes (Refined Model)")

    print("\nTraining Decision Tree model (baseline)...")
    dt_model = train_decision_tree(X_train, y_train)
    dt_accuracy = evaluate_model(dt_model, X_test, y_test, le, "Decision Tree (Baseline)")

    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Gaussian Naive Bayes Accuracy:  {nb_accuracy * 100:.2f}%")
    print(f"Bernoulli Naive Bayes Accuracy: {bnb_accuracy * 100:.2f}%")
    print(f"Decision Tree Accuracy:         {dt_accuracy * 100:.2f}%")

    # --- Generate evaluation visuals for Part D ---
    bnb_predictions = bnb_model.predict(X_test)

    # Confusion matrix (Bernoulli Naive Bayes - final model) with annotated counts
    cm = confusion_matrix(y_test, bnb_predictions)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Bernoulli Naive Bayes")
    plt.xlabel("Predicted Disease (encoded)")
    plt.ylabel("Actual Disease (encoded)")
    plt.colorbar()

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                plt.text(j, i, str(cm[i, j]),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black",
                          fontsize=8)

    plt.tight_layout()
    plt.savefig("docs/confusion_matrix_nb.png", dpi=150)
    plt.close()

    # Accuracy comparison bar chart (all three models)
    plt.figure(figsize=(7, 5))
    models = ["Gaussian NB\n(Initial)", "Bernoulli NB\n(Refined)", "Decision Tree\n(Baseline)"]
    accuracies = [nb_accuracy * 100, bnb_accuracy * 100, dt_accuracy * 100]
    colors = ["#90A4AE", "#2196F3", "#FF9800"]
    plt.bar(models, accuracies, color=colors)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison")
    plt.ylim(0, 105)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("docs/model_comparison.png", dpi=150)
    plt.close()

    print("\nVisuals saved to docs/ folder: confusion_matrix_nb.png, model_comparison.png")