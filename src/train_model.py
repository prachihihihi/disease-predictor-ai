import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import joblib
import os
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,
)
from augment_data import load_training, augment_dataset


def load_all_data():
    train_df = load_training("data/Training.csv")
    test_df = pd.read_csv("data/Testing.csv")
    test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]
    return train_df, test_df


def prepare_features(train_df, test_df, label_encoder=None):
    X_train = train_df.drop("prognosis", axis=1)
    y_train_raw = train_df["prognosis"]
    X_test = test_df.drop("prognosis", axis=1)
    y_test_raw = test_df["prognosis"]

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
    else:
        y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    return X_train, y_train, X_test, y_test, label_encoder


def run_cross_validation(X, y, models, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        results[name] = (scores.mean(), scores.std())
        print(f"{name:28s}  CV accuracy: {scores.mean()*100:6.2f}%  (+/- {scores.std()*100:.2f}%)")
    return results


def tune_logistic_regression(X, y):
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }
    grid = GridSearchCV(LogisticRegression(max_iter=2000), param_grid,
                         cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y)
    print(f"Best Logistic Regression params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_*100:.2f}%")
    return grid.best_estimator_


def tune_decision_tree(X, y):

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X, y)

    print(f"Best Decision Tree params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_*100:.2f}%")

    return grid.best_estimator_


def top_k_accuracy(model, X_test, y_test, k):
    proba = model.predict_proba(X_test)
    top_k_preds = np.argsort(proba, axis=1)[:, -k:]
    hits = [y_test[i] in top_k_preds[i] for i in range(len(y_test))]
    return np.mean(hits)


def evaluate_full(model, X_test, y_test, label_encoder, model_name):
    start = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    
    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
)

    recall = recall_score( 
        y_test,
        predictions,
        average="weighted",
        zero_division=0
        )
    
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
        )
    
    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")

    print(f"Prediction Time : {prediction_time:.6f} seconds")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")

    if hasattr(model, "predict_proba"):
        top3 = top_k_accuracy(model, X_test, y_test, 3)
        top5 = top_k_accuracy(model, X_test, y_test, min(5, len(label_encoder.classes_)))
        print(f"Top-3 Accuracy: {top3*100:.2f}%")
        print(f"Top-5 Accuracy: {top5*100:.2f}%")

    present_labels = sorted(set(y_test) | set(predictions))
    print("\nPer-class report:")
    print(classification_report(
        y_test, predictions,
        labels=present_labels,
        target_names=label_encoder.classes_[present_labels],
        zero_division=0
    ))

    return accuracy, precision, recall, f1, prediction_time, predictions


def plot_model_comparison(results_dict, out_path="docs/model_comparison.png"):
    """
    results_dict: { model_name: [clean_cv_acc, clean_test_acc, noisy_val_acc] }
    All values as fractions (0-1), converted to % for display.
    """
    stages = ["Clean 5-fold CV", "Clean test\n(noisy-trained)", "Noisy validation"]
    models = list(results_dict.keys())
    x = np.arange(len(stages))
    width = 0.8 / len(models)

    plt.figure(figsize=(9, 6))
    for i, (model_name, accs) in enumerate(results_dict.items()):
        offset = (i - (len(models) - 1) / 2) * width
        values = [a * 100 for a in accs]
        bars = plt.bar(x + offset, values, width, label=model_name)
        for bar, v in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%",
                      ha="center", fontsize=8)

    plt.xticks(x, stages)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    plt.title("Model Comparison Across Evaluation Stages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Model comparison chart saved to {out_path}")


def plot_confusion_matrix(y_test, predictions, label_encoder, out_path, title):
    present_labels = sorted(set(y_test) | set(predictions))
    cm = confusion_matrix(y_test, predictions, labels=present_labels)
    class_names = label_encoder.classes_[present_labels]

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted Disease")
    plt.ylabel("Actual Disease")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    plt.yticks(range(len(class_names)), class_names, fontsize=6)
    plt.colorbar()

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black", fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")

def plot_feature_importance(model, feature_names,
                            out_path="docs/feature_importance.png"):
        
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(indices)), importance[indices])
        
    plt.xticks(
        range(len(indices)),
        [feature_names[i].replace("_", " ") for i in indices],
        rotation=45,
        ha="right"
    )
        
    plt.ylabel("Importance")
        
    plt.title("Top 15 Most Important Symptoms")
        
    plt.tight_layout()
        
    plt.savefig(out_path, dpi=150)
        
    plt.close()
        
    print(f"Feature importance graph saved to {out_path}")


if __name__ == "__main__":
    train_df, test_df = load_all_data()
    X_train, y_train, X_test, y_test, le = prepare_features(train_df, test_df)

    print("\n" + "#" * 60)
    print("STEP 1: Stratified 5-fold CV on CLEAN training data")
    print("(This tells you if the dataset itself is trivially separable.)")
    print("#" * 60 + "\n")

    baseline_models = {
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    cv_results = run_cross_validation(X_train, y_train, baseline_models)

    print("\n" + "#" * 60)
    print("STEP 2: Hyperparameter tuning (Logistic Regression, Decision Tree)")
    print("#" * 60 + "\n")

    best_logreg = tune_logistic_regression(X_train, y_train)
    best_tree = tune_decision_tree(X_train, y_train)

    print("\n" + "#" * 60)
    print("STEP 3: Train final models on NOISE-AUGMENTED data")
    print("#" * 60 + "\n")

    aug_train_df = augment_dataset(train_df, drop_rate=0.2, add_rate=0.1, copies=3)
    X_aug, y_aug, _, _, _ = prepare_features(aug_train_df, test_df, label_encoder=le)
    print(f"Augmented training set: {X_aug.shape[0]} rows (from {X_train.shape[0]} original)")

    start = time.perf_counter()

    final_bnb = BernoulliNB()
    final_bnb.fit(X_aug, y_aug)

    bnb_train_time = time.perf_counter() - start

    start = time.perf_counter()

    final_tree = DecisionTreeClassifier(**best_tree.get_params())
    final_tree.fit(X_aug, y_aug)

    tree_train_time = time.perf_counter() - start


    start = time.perf_counter()
    
    final_logreg = LogisticRegression(**best_logreg.get_params())
    final_logreg.fit(X_aug, y_aug)
    
    logreg_train_time = time.perf_counter() - start

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_bnb, "models/bernoulli_nb.pkl")
    joblib.dump(final_tree, "models/decision_tree.pkl")
    joblib.dump(final_logreg, "models/logistic_regression.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    
    print("\nModels saved successfully.")

    print("\n" + "#" * 60)
    print("STEP 4: Evaluation on ORIGINAL clean Testing.csv")
    print("#" * 60 + "\n")

    acc_bnb, prec_bnb, rec_bnb, f1_bnb, pred_time_bnb, pred_bnb = evaluate_full(final_bnb, X_test, y_test, le,
                                        "Bernoulli NB (trained on noisy data)")
    print(f"Training Time : {bnb_train_time:.6f} seconds")

    acc_tree, prec_tree, rec_tree, f1_tree, pred_time_tree, pred_tree = evaluate_full(final_tree, X_test, y_test,le,
                                        "Decision Tree (trained on noisy data)")
    print(f"Training Time : {tree_train_time:.6f} seconds")
    
    acc_logreg, prec_logreg, rec_logreg, f1_logreg, pred_time_logreg, pred_logreg = evaluate_full(final_logreg, X_test, y_test, le,
                                             "Logistic Regression (trained on noisy data)")
    print(f"Training Time : {logreg_train_time:.6f} seconds")

    plot_confusion_matrix(y_test, pred_bnb, le, "docs/confusion_matrix_bnb.png",
                           "Confusion Matrix - Bernoulli NB (noise-trained)")
    plot_confusion_matrix(y_test, pred_tree, le, "docs/confusion_matrix_tree.png",
                           "Confusion Matrix - Decision Tree (noise-trained)")
    plot_confusion_matrix(y_test, pred_logreg, le, "docs/confusion_matrix_logreg.png",
                           "Confusion Matrix - Logistic Regression (noise-trained)")
    
    plot_feature_importance(
        final_tree,
        X_train.columns
        )
    
    print("\n" + "#" * 60)
    print("STEP 5: Evaluation on a NOISY, held-out validation split")
    print("(The gap vs STEP 1 is your main evidence of real robustness.)")
    print("#" * 60 + "\n")

    noisy_full = augment_dataset(train_df, drop_rate=0.3, add_rate=0.15, copies=1)
    X_noisy, y_noisy, _, _, _ = prepare_features(noisy_full, test_df, label_encoder=le)
    _, X_noisy_val, _, y_noisy_val = train_test_split(
        X_noisy, y_noisy, test_size=0.3, stratify=y_noisy, random_state=42
    )

    acc_bnb_noisy, _, _, _, _, _ = evaluate_full(
        final_bnb, 
        X_noisy_val, 
        y_noisy_val, 
        le,
        "Bernoulli NB on noisy validation split"
    )
    
    acc_logreg_noisy, _, _, _, _, _ = evaluate_full(
        final_logreg,
        X_noisy_val,
        y_noisy_val,
        le,
        "Logistic Regression on noisy validation split"
    )
    
    acc_tree_noisy, _, _, _, _, _ = evaluate_full(
        final_tree,
        X_noisy_val,
        y_noisy_val,
        le,
        "Decision Tree on noisy validation split"
    )

    print("\n" + "#" * 60)
    print("Generating model comparison chart")
    print("#" * 60 + "\n")

    comparison_results = {
        "Bernoulli NB": [cv_results["Bernoulli Naive Bayes"][0], acc_bnb, acc_bnb_noisy],
        "Decision Tree": [cv_results["Decision Tree"][0], acc_tree, acc_tree_noisy],
        "Logistic Regression": [cv_results["Logistic Regression"][0], acc_logreg, acc_logreg_noisy],
    }
    plot_model_comparison(comparison_results)

    print("\nDone. Compare STEP 1 (clean CV) vs STEP 5 (noisy validation) accuracy —")
    print("that gap is the central evidence for your report.")

    model_scores = {
    "Bernoulli NB": acc_bnb,
    "Decision Tree": acc_tree,
    "Logistic Regression": acc_logreg
  }
    best_model_name = max(
        model_scores,
        key=model_scores.get
        )
    
    best_model = {
        "Bernoulli NB": final_bnb,
        "Decision Tree": final_tree,
        "Logistic Regression": final_logreg
        }[best_model_name]
    
    print("\nBest Model:")
    print(best_model_name)
    
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(
        best_model,
        "models/best_model.pkl"
        )


    joblib.dump(
        X_train.columns.tolist(),
        "models/symptom_list.pkl"
    )


    print("Best model saved.")