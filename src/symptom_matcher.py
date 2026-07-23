"""
symptom_matcher.py

WHY THIS EXISTS:
The original CLI required users to type symptom names EXACTLY as they appear
in the dataset (e.g. "burning_micturition" instead of "it burns when I pee").
This module fixes that.

Three layers of matching, tried in order:
 1. Exact match (after normalizing spaces/case to underscores)
 2. Synonym dictionary — common lay phrases mapped to dataset symptom names
 3. Fuzzy string matching (difflib) — catches typos and near-misses,
    e.g. "vomitting" -> "vomiting"

If a fuzzy match is uncertain, the user is asked to confirm with yes/no
rather than the system silently guessing wrong.

This does NOT alter the prediction model itself — it only changes how raw
user input gets converted into the binary symptom vector the model expects.
"""

import difflib


SYNONYM_MAP = {
    "throwing up": "vomiting", "throw up": "vomiting", "puking": "vomiting",
    "feeling sick": "nausea", "feel sick": "nausea",
    "stomach ache": "stomach_pain", "tummy ache": "stomach_pain", "stomach pain": "stomach_pain",
    "belly ache": "belly_pain",
    "tired": "fatigue", "feeling tired": "fatigue", "exhausted": "fatigue", "no energy": "lethargy",
    "high temperature": "high_fever", "temperature": "high_fever", "low fever": "mild_fever",
    "body ache": "muscle_pain", "body aches": "muscle_pain",
    "joint ache": "joint_pain", "aching joints": "joint_pain",
    "yellow eyes": "yellowing_of_eyes", "yellow skin": "yellowish_skin",
    "dark pee": "dark_urine", "dark urine": "dark_urine",
    "blood in stool": "bloody_stool", "blood in poop": "bloody_stool",
    "chest tightness": "chest_pain", "tight chest": "chest_pain",
    "dizzy": "dizziness", "feeling dizzy": "dizziness",
    "cant smell": "loss_of_smell", "can't smell": "loss_of_smell",
    "losing weight": "weight_loss", "gaining weight": "weight_gain", "putting on weight": "weight_gain",
    "cold hands": "cold_hands_and_feets", "cold feet": "cold_hands_and_feets",
    "anxious": "anxiety", "feeling anxious": "anxiety",
    "depressed": "depression", "feeling depressed": "depression",
    "irritable": "irritability",
    "sore throat": "throat_irritation", "throat pain": "throat_irritation",
    "itchy skin": "itching", "skin itching": "itching",
    "rash": "skin_rash",
    "pimples": "pus_filled_pimples", "acne": "pus_filled_pimples",
    "stiff neck": "stiff_neck",
    "constipated": "constipation",
    "diarrhea": "diarrhoea", "loose motion": "diarrhoea", "loose motions": "diarrhoea",
    "swollen legs": "swollen_legs",
    "fast heartbeat": "fast_heart_rate", "racing heart": "fast_heart_rate",
    "blurry vision": "blurred_and_distorted_vision", "blurred vision": "blurred_and_distorted_vision",
    "no appetite": "loss_of_appetite", "not hungry": "loss_of_appetite",
    "very hungry": "excessive_hunger", "always hungry": "excessive_hunger",
    "burning pee": "burning_micturition", "burning when i pee": "burning_micturition",
    "painful urination": "burning_micturition",
    "frequent urination": "polyuria", "peeing a lot": "polyuria", "peeing frequently": "polyuria",
    "runny nose": "runny_nose", "stuffy nose": "congestion", "blocked nose": "congestion",
    "sneezing": "continuous_sneezing",
    "shaking": "shivering", "trembling": "shivering", "cold chills": "chills",
    "back ache": "back_pain", "neck ache": "neck_pain",
    "weak legs": "weakness_in_limbs", "weak arms": "weakness_in_limbs",
    "slurred words": "slurred_speech", "hard to speak": "slurred_speech",
    "knee ache": "knee_pain", "hip ache": "hip_joint_pain",
    "swollen joints": "swelling_joints", "stiff joints": "movement_stiffness",
    "spinning": "spinning_movements",
    "feeling off balance": "loss_of_balance", "unsteady": "unsteadiness",
    "watery eyes": "watering_from_eyes", "red eyes": "redness_of_eyes",
    "puffy eyes": "puffy_face_and_eyes", "puffy face": "puffy_face_and_eyes",
    "swollen thyroid": "enlarged_thyroid",
    "brittle nails": "brittle_nails", "weak nails": "brittle_nails",
    "cough": "cough", "coughing": "cough",
    "phlegm": "phlegm", "mucus": "phlegm",
    "sinus pain": "sinus_pressure", "sinus pressure": "sinus_pressure",
    "swollen glands": "swelled_lymph_nodes", "swollen lymph nodes": "swelled_lymph_nodes",
    "cramps": "cramps", "muscle cramps": "cramps",
    "bruising easily": "bruising", "easy bruising": "bruising",
    "obese": "obesity", "overweight": "obesity",
    "breathless": "breathlessness", "short of breath": "breathlessness", "trouble breathing": "breathlessness",
    "sweating a lot": "sweating", "excessive sweating": "sweating",
    "dehydrated": "dehydration",
    "indigestion": "indigestion",
    "heartburn": "acidity", "acid reflux": "acidity",
    "headache": "headache", "head pain": "headache",
    "coma": "coma",
    "confused": "altered_sensorium", "confusion": "altered_sensorium",
    "mood swings": "mood_swings",
    "restless": "restlessness",
    "malaise": "malaise", "feeling unwell": "malaise",
    "skin peeling": "skin_peeling", "peeling skin": "skin_peeling",
    "small dents in nails": "small_dents_in_nails",
    "blister": "blister", "blisters": "blister",
    "red spots": "red_spots_over_body", "red spots on body": "red_spots_over_body",
}


def normalize(text):
    return text.strip().lower().replace(" ", "_")


def match_symptom(user_phrase, symptom_list, cutoff=0.6):
    """
    Returns (matched_symptom_or_candidates, confidence_level)
    confidence_level: "exact", "synonym", "fuzzy", "ambiguous", "none"
    "ambiguous" returns a LIST of candidates — caller should disambiguate.
    """
    raw = user_phrase.strip().lower()
    norm = normalize(raw)
    normalized_lookup = {normalize(s): s for s in symptom_list}

    if norm in normalized_lookup:
        return normalized_lookup[norm], "exact"

    if raw in SYNONYM_MAP:
        candidate = SYNONYM_MAP[raw]
        if candidate in symptom_list:
            return candidate, "synonym"

    close = difflib.get_close_matches(norm, list(normalized_lookup.keys()), n=3, cutoff=cutoff)
    if len(close) == 1:
        return normalized_lookup[close[0]], "fuzzy"
    elif len(close) > 1:
        return [normalized_lookup[c] for c in close], "ambiguous"

    return None, "none"


def resolve_symptom_list(raw_phrases, symptom_list, confirm_callback=None):
    confirmed = []
    unresolved = []

    for phrase in raw_phrases:
        result, confidence = match_symptom(phrase, symptom_list)

        if confidence in ("exact", "synonym"):
            confirmed.append(result)
        elif confidence == "fuzzy":
            choice = confirm_callback(phrase, [result]) if confirm_callback else None
            (confirmed if choice else unresolved).append(choice or phrase)
        elif confidence == "ambiguous":
            choice = confirm_callback(phrase, result) if confirm_callback else None
            (confirmed if choice else unresolved).append(choice or phrase)
        else:
            unresolved.append(phrase)

    return list(dict.fromkeys(confirmed)), unresolved