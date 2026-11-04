import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)

    prediction = accuracy_score(y_test, model.predict(X_test))
    cols = [c for c in X.columns if X[c].nunique() > 10]
    rng = np.random.default_rng(42)
    noise = rng.normal(size = (len(X_test), len(cols)))
    results = {}
    predictions = {}

    for amount in [0.05, 0.10, 0.20]:
        changed = X_test.copy()
        changed[cols] = changed[cols] + noise * X_train[cols].std().replace(0, 1).values * amount
        predictions[amount] = model.predict(changed)
        results[amount] = accuracy_score(y_test, predictions[amount])

    worst = min(results, key = results.get)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions[worst], labels = [0, 1]).ravel()

    report = f"""
        1. Experimental Setups

        Model  :  Random Forest
        Dataset  :  heart.csv
        Training data  :  80%
        Testing data  :  20%
        Split method  :  stratified
        Random state  :  42
        Testing procedure  :  Feature perturbations of 0.05, 0.10 and 0.20
        standard deviations on numerical features with more than 10 unique values.
        Features tested  :  {", ".join(cols)}

        2. Performance Trends

        Prediction Accuracy  :  {prediction:.3f}
        Accuracy at 0.05  :  {results[0.05]:.3f}
        Accuracy at 0.10  :  {results[0.10]:.3f}
        Accuracy at 0.20  :  {results[0.20]:.3f}
        Accuracy Drop in Worst Test  :  {prediction - results[worst]:.3f}

        3. Failure Modes

        Worst Testing Condition  :  {worst}
        True Positives  :  {tp}
        True Negatives  :  {tn}
        False Positives  :  {fp}
        False Negatives  :  {fn}

        4. Recommendations for Improving Model Resilience

        - Train with more varied and representative data.
        - Test the model using realistic feature changes.
        - Improve feature preprocessing and data quality.
        - Tune model parameters to improve stability.
        - Monitor model performance regularly.
        """

    st.subheader("Robustness Testing Technical Report")
    st.text(report)