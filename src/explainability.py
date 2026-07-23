"""
explainability.py

Shows which symptoms drove a given prediction (general disease profile and
per-patient explanation), using BernoulliNB's feature_log_prob_.

Run: python explainability.py
"""

import numpy as np


def explain_disease_profile(model, label_encoder, symptom_cols, disease_name, top_n=10):
    class_idx = list(label_encoder.classes_).index(disease_name)
    log_probs = model.feature_log_prob_[class_idx]

    ranking = sorted(zip(symptom_cols, log_probs), key=lambda x: x[1], reverse=True)
    top_symptoms = ranking[:top_n]

    print(f"\nSymptoms most associated with '{disease_name}' (general profile):")
    for symptom, log_p in top_symptoms:
        prob = np.exp(log_p)
        print(f"  {symptom:30s}  P(present | disease) = {prob:.3f}")

    return top_symptoms


def explain_this_prediction(model, label_encoder, symptom_cols,
                             entered_symptoms, predicted_disease, top_n=10):
    class_idx = list(label_encoder.classes_).index(predicted_disease)
    log_probs = model.feature_log_prob_[class_idx]

    contributions = []
    for symptom in entered_symptoms:
        if symptom in symptom_cols:
            idx = symptom_cols.index(symptom)
            contributions.append((symptom, log_probs[idx]))

    contributions.sort(key=lambda x: x[1], reverse=True)
    top = contributions[:top_n]

    print(f"\nWhy the model predicted '{predicted_disease}' for this patient:")
    for symptom, log_p in top:
        prob = np.exp(log_p)
        print(f"  {symptom:30s}  weight = {log_p:.3f}  (P(present|disease)={prob:.3f})")

    return top


if __name__ == "__main__":
    from train_model import load_all_data, prepare_features
    from sklearn.naive_bayes import BernoulliNB

    train_df, test_df = load_all_data()
    X_train, y_train, X_test, y_test, le = prepare_features(train_df, test_df)

    model = BernoulliNB()
    model.fit(X_train, y_train)

    symptom_cols = X_train.columns.tolist()

    example_disease = le.classes_[0]
    explain_disease_profile(model, le, symptom_cols, example_disease)

    sample_symptoms = symptom_cols[:4]
    predicted = le.classes_[model.predict(
        [[1 if s in sample_symptoms else 0 for s in symptom_cols]]
    )[0]]
    explain_this_prediction(model, le, symptom_cols, sample_symptoms, predicted)