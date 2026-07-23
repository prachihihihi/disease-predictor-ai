import joblib
import os

files = [
    "bernoulli_nb.pkl",
    "decision_tree.pkl",
    "logistic_regression.pkl",
    "best_model.pkl",
    "label_encoder.pkl",
    "symptom_list.pkl"
]

for file in files:
    path = os.path.join("models", file)

    print("\nChecking:", file)

    try:
        obj = joblib.load(path)

        print("✅ Loaded successfully")
        print("Type:", type(obj))

        if hasattr(obj, "classes_"):
            print("Number of classes:", len(obj.classes_))

        if isinstance(obj, list):
            print("Length:", len(obj))
            print("First items:", obj[:5])

    except Exception as e:
        print("❌ Failed")
        print(type(e).__name__, e)