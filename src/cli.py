import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

def load_and_train():
    train_df = pd.read_csv("data/Training.csv")
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

    X_train = train_df.drop("prognosis", axis=1)
    y_train = train_df["prognosis"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    model = GaussianNB()
    model.fit(X_train, y_encoded)

    symptom_list = X_train.columns.tolist()
    return model, le, symptom_list

def get_user_symptoms(symptom_list):
    print("\nAvailable symptoms (type the exact name, comma-separated):")
    print(", ".join(symptom_list[:15]) + " ... (and more)")
    print("\nTip: type 'list' to see all symptoms, or 'quit' to exit.\n")

    while True:
        user_input = input("Enter your symptoms: ").strip()

        if user_input.lower() == "quit":
            return None

        if user_input.lower() == "list":
            print("\n" + ", ".join(symptom_list) + "\n")
            continue

        entered = [s.strip().lower().replace(" ", "_") for s in user_input.split(",")]
        valid = [s for s in entered if s in symptom_list]
        invalid = [s for s in entered if s not in symptom_list]

        if invalid:
            print(f"\nNot recognised: {invalid}. Try 'list' to see valid symptom names.\n")
            continue

        if not valid:
            print("\nNo valid symptoms entered. Try again.\n")
            continue

        return valid

def predict_disease(model, le, symptom_list, entered_symptoms):
    input_vector = [1 if symptom in entered_symptoms else 0 for symptom in symptom_list]
    input_df = pd.DataFrame([input_vector], columns=symptom_list)

    probabilities = model.predict_proba(input_df)[0]
    disease_names = le.classes_

    results = sorted(zip(disease_names, probabilities), key=lambda x: x[1], reverse=True)
    return results[:3]  # top 3 predictions

def main():
    print("=" * 55)
    print("   AI Disease Predictor (Naive Bayes)")
    print("=" * 55)

    model, le, symptom_list = load_and_train()

    while True:
        entered_symptoms = get_user_symptoms(symptom_list)

        if entered_symptoms is None:
            print("\nGoodbye!")
            break

        top_predictions = predict_disease(model, le, symptom_list, entered_symptoms)

        print("\n--- Top Predictions ---")
        for disease, prob in top_predictions:
            print(f"  {disease}: {prob * 100:.2f}% confidence")
        print()

if __name__ == "__main__":
    main()