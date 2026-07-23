# Disease Predictor AI

An AI-based system that predicts probable diseases based on user-input symptoms.

The system compares multiple machine learning approaches, including Bernoulli Naive Bayes, Decision Tree, and Logistic Regression. The final prediction model is selected based on evaluation performance and used for interactive disease prediction.

The application supports natural language symptom input, symptom matching using synonyms and fuzzy matching, adaptive follow-up questions, confidence scoring, explainability, and patient prediction reports.


## Features

- Disease prediction from user symptoms
- Natural language symptom recognition
- Synonym and fuzzy symptom matching
- Adaptive yes/no follow-up questions
- Multiple machine learning model comparison
- Confidence score generation
- Prediction explanation using symptom importance
- Automatic patient prediction report generation


## Machine Learning Approach

The project evaluates different supervised learning approaches:

- Bernoulli Naive Bayes
- Decision Tree
- Logistic Regression

The best-performing model is saved as: models/best_model.pkl

and is used by the interactive prediction system.


## How to Run

### 1. Install dependencies: pip3 install -r requirements.txt

### 2. Train and evaluate models: python3 src/train_model.py

### 3. Check saved models: python3 src/check_models.py

### 4. Run the interactive predictor: python3 src/main.py

### 5. Generate explainability output: src/explainability.py


## Machine Learning Approach

The project evaluates different supervised learning approaches:

- Bernoulli Naive Bayes
- Decision Tree
- Logistic Regression

The models are compared using evaluation metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- Top-k prediction accuracy
- Confusion matrices

The best-performing model is saved as: models/best_model.pkl
and is used by the interactive prediction system.


## Disclaimer

This system is developed for educational purposes only.

It provides possible disease predictions from symptoms and should not replace professional medical diagnosis or emergency healthcare services.


## Module
ST5001CMD - Artificial Intelligence


## Author
Prachi Silwal