import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder


def load_and_train():
    train_df = pd.read_csv("data/Training.csv")
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

    X_train = train_df.drop("prognosis", axis=1)
    y_train = train_df["prognosis"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    model = BernoulliNB()
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

        raw_entered = [
            s.strip().lower().replace(" ", "_")
            for s in user_input.split(",")
        ]

        entered = list(dict.fromkeys(raw_entered))

        if len(raw_entered) != len(entered):
            print("\nDuplicate symptoms detected. Duplicates have been removed.\n")

        valid = [s for s in entered if s in symptom_list]
        invalid = [s for s in entered if s not in symptom_list]

        if invalid:
            print(f"\nNot recognised: {invalid}")
            print("Type 'list' to see all valid symptom names.\n")
            continue

        if not valid:
            print("\nNo valid symptoms entered. Try again.\n")
            continue

        return valid


def predict_disease(model, le, symptom_list, entered_symptoms):
    input_vector = [
        1 if symptom in entered_symptoms else 0
        for symptom in symptom_list
    ]

    input_df = pd.DataFrame([input_vector], columns=symptom_list)

    probabilities = model.predict_proba(input_df)[0]
    disease_names = le.classes_

    results = sorted(
        zip(disease_names, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    return results[:3]


def main():

    print("=" * 60)
    print("        AI Disease Prediction System")
    print("=" * 60)
    print("DISCLAIMER")
    print("-" * 60)
    print("This application is for educational purposes only.")
    print("Predictions are generated using a machine learning model")
    print("and should NOT be considered a medical diagnosis.")
    print("Always consult a qualified healthcare professional.")
    print("=" * 60)

    input("Press Enter to continue...")

    model, le, symptom_list = load_and_train()

    while True:

        entered_symptoms = get_user_symptoms(symptom_list)

        if entered_symptoms is None:
            print("\nThank you for using the AI Disease Prediction System.")
            break

        # Warning if too few symptoms are entered
        if len(entered_symptoms) < 3:
            print("\nWARNING")
            print("-" * 60)
            print("Only a small number of symptoms were entered.")
            print("Predictions based on limited information may")
            print("be less reliable.")
            print("-" * 60)

        top_predictions = predict_disease(
            model,
            le,
            symptom_list,
            entered_symptoms
        )

        print("\nTop 3 Predicted Diseases")
        print("-" * 60)

        for disease, probability in top_predictions:
            print(f"{disease}: {probability * 100:.2f}% confidence")

        print()


if __name__ == "__main__":
    main()
    