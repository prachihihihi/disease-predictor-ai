import joblib
import pandas as pd
import os
from disease_info import DISEASE_INFO
from symptom_matcher import match_symptom
from datetime import datetime

MAX_FOLLOWUP_QUESTIONS = 5
CONFIDENCE_STOP_THRESHOLD = 0.60   # stop auto-asking once top prediction is this confident
DISPLAY_TOP_N = 3                  # user-facing cap — matches differential-diagnosis convention
DISPLAY_MIN_CONFIDENCE = 0.05      # suppress predictions below this — not genuine candidates

def print_session_summary(patient, symptoms, predictions, model):    
    print("\n")
    print("=" * 60)
    print("PATIENT SUMMARY")
    print("=" * 60)

    print(f"Patient Name : {patient['name']}")
    print(f"Age          : {patient['age']}")
    print(f"Gender       : {patient['gender']}")
    print(f"Date         : {patient['date']}")

    print("\nSymptoms Entered")
    print()
    print(f"Total Symptoms Used : {len(symptoms)}")
    print("-" * 60)

    for symptom in symptoms:
        print(f"✓ {symptom.replace('_', ' ').capitalize()}")

    print()
    print(f"Total Symptoms Recorded : {len(symptoms)}")

    print("\nTop Predictions")
    print("-" * 60)

    for i, (disease, probability, _) in enumerate(predictions, start=1):
        print(
        f"{i}. {disease.title():<30} "
        f"{probability*100:6.2f}%"
    )
    best_disease = predictions[0][0]

    print("\nMatched Symptoms")
    print("-" * 60)

    for symptom in symptoms:
        print(f"✓ {symptom.replace('_', ' ').title()} contributed to the prediction.")
    
    if best_disease in DISEASE_INFO:
        info = DISEASE_INFO[best_disease]
        print("\nPossible Causes")
        print("-" * 60)
        
        for cause in info["causes"]:
            print(f"• {cause}")

        print("\nRecommended Self Care")
        print("-" * 60)

        for tip in info["care"]:
            print(f"✓ {tip}")

        print("\nEmergency Advice")
        print("-" * 60)

        if info["emergency"]:
            print("⚠ Seek immediate medical attention if symptoms become severe.")
        else:
            print("No immediate emergency warning based on this prediction.")

        top_probability = predictions[0][1]

        print("\nPrediction Reliability")
        print("-" * 60)

        if top_probability >= 0.80:
            print("HIGH")
            print("The entered symptoms strongly match one disease profile.")

        elif top_probability >= 0.60:
            print("MODERATE")
            print("The prediction is reasonable, but additional symptoms or")
            print("medical assessment may improve confidence.")

        else:
            print("LOW")
            print("The entered symptoms match several diseases.")
            print("More symptoms are recommended before relying on this prediction.")

        print(f"\nConfidence Score : {top_probability*100:.2f}%")
        
        print("\nPrediction Generated At")
        print("-" * 60)
        print(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))

        print("\nPrediction Model")
        print("-" * 60)
        print(model.__class__.__name__)

        print("\nDISCLAIMER")
        print("-" * 60)
        print("This system is intended to assist with symptom assessment")
        print("and should never replace professional medical diagnosis")
        print("or emergency healthcare services.")

        print("=" * 60)

def save_patient_report(patient, symptoms, predictions):

    os.makedirs("reports", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"reports/{patient['name']}_{timestamp}.txt"

    with open(filename, "w") as f:

        f.write("AI Disease Prediction System\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Patient Name : {patient['name']}\n")
        f.write(f"Age          : {patient['age']}\n")
        f.write(f"Gender       : {patient['gender']}\n")
        f.write(f"Date         : {patient['date']}\n\n")

        f.write("Symptoms\n")
        f.write("-" * 60 + "\n")

        for symptom in symptoms:
            f.write(f"• {symptom.replace('_',' ').title()}\n")

        f.write("\nTop Predictions\n")
        f.write("-" * 60 + "\n")

        for i, (disease, probability, _) in enumerate(predictions, start=1):
            f.write(f"{i}. {disease.title():<30} {probability*100:.2f}%\n")

        top_probability = predictions[0][1]

        f.write("\nPrediction Reliability\n")
        f.write("-" * 60 + "\n")

        if top_probability >= 0.80:
            f.write("HIGH\n")
        elif top_probability >= 0.60:
            f.write("MODERATE\n")
        else:
            f.write("LOW\n")

        f.write("\nDISCLAIMER\n")
        f.write("-" * 60 + "\n")
        f.write("This report is generated using a machine learning model.\n")
        f.write("It is intended for educational purposes only.\n")
        f.write("Consult a qualified healthcare professional.\n")

    print(f"\nPatient report saved successfully:\n{filename}")

def register_patient():

    print("\n" + "=" * 60)
    print("               PATIENT REGISTRATION")
    print("=" * 60)

    while True:
        name = input("Patient Name : ").strip()
        if name:
            break
        print("Name cannot be empty.")

    while True:
        age = input("Age          : ").strip()

        if age.isdigit() and 0 < int(age) < 120:
            age = int(age)
            break

        print("Please enter a valid age.")

    while True:
        gender = input("Gender (M/F/O): ").strip().upper()

        if gender in ["M", "F", "O"]:
            gender = {
                "M": "Male",
                "F": "Female",
                "O": "Other"
            }[gender]
            break

        print("Please enter M, F or O.")

    print("=" * 60)

    return {
        "name": name,
        "age": age,
        "gender": gender,
        "date": datetime.now().strftime("%d-%m-%Y %H:%M")
    }

def load_model():

    model = joblib.load(
        "models/best_model.pkl"
    )

    le = joblib.load(
        "models/label_encoder.pkl"
    )

    symptom_list = joblib.load(
        "models/symptom_list.pkl"
    )

    return model, le, symptom_list

def confirm_match(phrase, candidates):
   
    if len(candidates) == 1:
        answer = input(f"  You mentioned '{phrase}' — did you mean '{candidates[0]}'? (y/n): ").strip().lower()
        return candidates[0] if answer.startswith("y") else None

    print(f"  '{phrase}' could match a few things:")
    for i, c in enumerate(candidates, 1):
        print(f"    {i}. {c}")
    print(f"    0. None of these")
    while True:
        choice = input("  Enter a number: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print("  Please enter a valid number.")


def resolve_phrases_to_symptoms(raw_phrases, symptom_list):
    
    confirmed = []
    for phrase in raw_phrases:
        result, confidence = match_symptom(phrase, symptom_list)

        if confidence == "exact" or confidence == "synonym":
            confirmed.append(result)
            print(f"  Recognised '{phrase}' as '{result}'")
        elif confidence == "fuzzy":
            chosen = confirm_match(phrase, [result])
            if chosen:
                confirmed.append(chosen)
        elif confidence == "ambiguous":
            chosen = confirm_match(phrase, result)
            if chosen:
                confirmed.append(chosen)
        else:
            print(f"  Could not recognise '{phrase}' — skipping it.")

    return list(dict.fromkeys(confirmed))


def get_initial_symptoms(symptom_list):
    print("\nDescribe your symptoms in your own words, separated by commas.")
    print("(e.g. \"throwing up, high temperature, tummy ache\")")
    
    while True:
        user_input = input("Your symptoms: ").strip()
        if user_input.lower() == "quit":
            return None

        raw_phrases = [p.strip() for p in user_input.split(",") if p.strip()]
        if not raw_phrases:
            print("Please enter at least one symptom, or type 'quit' to exit.\n")
            continue

        confirmed = resolve_phrases_to_symptoms(raw_phrases, symptom_list)

        if not confirmed:
            print("\nNone of that could be matched to a known symptom. Try rephrasing.\n")
            continue

        if len(confirmed) < 3:
            print("\nNote: only a few symptoms were recognised. The follow-up")
            print("questions below will help narrow things down.\n")

        return confirmed


def ask_anything_else(symptom_list, symptom_vector, asked):
    
    answer = input("\nIs there anything else about your symptoms you'd like to mention? (y/n): ").strip().lower()
    if not answer.startswith("y"):
        return symptom_vector, asked, False

    user_input = input("Please describe it (comma-separated if more than one): ").strip()
    raw_phrases = [p.strip() for p in user_input.split(",") if p.strip()]
    if not raw_phrases:
        return symptom_vector, asked, False

    newly_confirmed = resolve_phrases_to_symptoms(raw_phrases, symptom_list)
    newly_confirmed = [s for s in newly_confirmed if s not in asked]

    if not newly_confirmed:
        print("  Nothing new was added.")
        return symptom_vector, asked, False

    for s in newly_confirmed:
        idx = symptom_list.index(s)
        symptom_vector[idx] = 1
        asked.add(s)

    print(f"  Added: {', '.join(newly_confirmed)}")
    return symptom_vector, asked, True


def get_candidate_diseases(model, le, symptom_list, symptom_vector, top_n=3):
    input_df = pd.DataFrame([symptom_vector], columns=symptom_list)
    probabilities = model.predict_proba(input_df)[0]
    ranking = sorted(zip(le.classes_, probabilities, range(len(le.classes_))),
                      key=lambda x: x[1], reverse=True)
    return ranking[:top_n]


def pick_next_question(model, symptom_list, symptom_vector, asked, candidate_class_indices):

    scores = model.feature_importances_

    ranking = sorted(
        [(s, scores[i]) for i, s in enumerate(symptom_list)
         if s not in asked and symptom_vector[i] == 0],
        key=lambda x: x[1],
        reverse=True
    )

    return ranking[0][0] if ranking else None


def ask_one_question(model, le, symptom_list, symptom_vector, asked, top_n=3):
    
    candidates = get_candidate_diseases(model, le, symptom_list, symptom_vector, top_n=top_n)
    candidate_class_indices = [c[2] for c in candidates]
    next_symptom = pick_next_question(model, symptom_list, symptom_vector, asked, candidate_class_indices)

    if next_symptom is None:
        print("No further distinguishing symptoms to ask about.")
        return symptom_vector, asked, False

    readable = next_symptom.replace("_", " ")
    answer = input(f"Do you also have: {readable}? (y/n): ").strip().lower()
    asked.add(next_symptom)

    if answer.startswith("y"):
        idx = symptom_list.index(next_symptom)
        symptom_vector[idx] = 1

    return symptom_vector, asked, True


def run_followup_questions(model, le, symptom_list, symptom_vector, confirmed_symptoms):
    asked = set(confirmed_symptoms)
    questions_asked = 0

    while questions_asked < MAX_FOLLOWUP_QUESTIONS:
        candidates = get_candidate_diseases(model, le, symptom_list, symptom_vector, top_n=3)
        top_prob = candidates[0][1]

        if top_prob >= CONFIDENCE_STOP_THRESHOLD:
            break

        symptom_vector, asked, asked_something = ask_one_question(
            model, le, symptom_list, symptom_vector, asked, top_n=3
        )
        if not asked_something:
            break
        questions_asked += 1

    return symptom_vector, asked


def show_predictions(model, le, symptom_list, symptom_vector,
                      top_n=DISPLAY_TOP_N, min_confidence=DISPLAY_MIN_CONFIDENCE):
    
    all_candidates = get_candidate_diseases(model, le, symptom_list, symptom_vector, top_n=top_n)
    shown = [c for c in all_candidates if c[1] >= min_confidence]
    if not shown:
        shown = all_candidates[:1]

    print(f"\nTop Predicted Disease{'s' if len(shown) > 1 else ''}")
    print("-" * 60)
    for disease, probability, _ in shown:
        print(f"{disease}: {probability * 100:.2f}% confidence")
    if len(shown) < len(all_candidates):
        n_hidden = len(all_candidates) - len(shown)
        print(f"({n_hidden} other low-confidence possibilit{'y' if n_hidden == 1 else 'ies'} "
              f"below {min_confidence*100:.0f}% not shown)")
    print()
    return all_candidates


def main():
    print("=" * 60)
    print("        AI Disease Prediction System (Interactive)")
    print("=" * 60)
    print("DISCLAIMER")
    print("-" * 60)
    print("This application is for educational purposes only.")
    print("Predictions are generated using a machine learning model")
    print("and should NOT be considered a medical diagnosis.")
    print("Always consult a qualified healthcare professional.")
    print("=" * 60)

    input("Press Enter to continue...")
    patient = register_patient()

    model, le, symptom_list = load_model()

    while True:
        confirmed = get_initial_symptoms(symptom_list)
        if confirmed is None:
            print("\nThank you for using the AI Disease Prediction System.")
            break

        symptom_vector = [1 if s in confirmed else 0 for s in symptom_list]

        print("\nA few quick yes/no questions to narrow this down:\n")
        symptom_vector, asked = run_followup_questions(
            model, le, symptom_list, symptom_vector, confirmed
            )
        
        predictions = show_predictions( 
            model,
            le,
            symptom_list,
            symptom_vector
            )
        
        while True:
            refine = input(
                "Would you like to answer one more question to refine this? (y/n): "
                ).strip().lower()
            
            if refine.startswith("y"):
                symptom_vector, asked, asked_something = ask_one_question(
                    model,
                    le,
                    symptom_list,
                    symptom_vector,
                    asked,
                    top_n=5
                    )
                
                if not asked_something:
                    break
                
                predictions = show_predictions(
                    model,
                    le,
                    symptom_list,
                    symptom_vector)
                
            else:
                break
            
        all_symptoms = [
            symptom_list[i]
            for i, value in enumerate(symptom_vector)
            if value == 1
            ]
        
        print_session_summary(
            patient,
            all_symptoms,
            predictions,
            model
            
            )
        
        save_patient_report(
            patient,
            all_symptoms,
            predictions
            )
        
        restart = input("Check a different set of symptoms instead? (y/n): ").strip().lower()

        if not restart.startswith("y"):
            print("\nThank you for using the AI Disease Prediction System.")
            break


if __name__ == "__main__":
    main()