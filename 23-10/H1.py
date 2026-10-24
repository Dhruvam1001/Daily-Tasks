import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42, stratify = y_train)
    
    models = {
        "Decision tree": DecisionTreeClassifier(max_depth = 5, random_state = 42),
        "SVM": make_pipeline(StandardScaler(), SVC(probability = True)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5))
    }
    
    weights = {}
    test_preds = {}
    test_probs = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        weights[name] = accuracy_score(y_val, val_pred)
        test_preds[name] = model.predict(X_test)
        test_probs[name] = model.predict_proba(X_test)

    classes = np.array(sorted(y.unique()))
    hard_scores = np.zeros((len(X_test), len(classes)))

    for name in models:
        for i, cls in enumerate(classes):
            hard_scores[:, i] += weights[name] * (test_preds[name] == cls)

    hard_pred = classes[np.argmax(hard_scores, axis=1)]
    total_weight = sum(weights.values())
    soft_probs = sum(weights[name] * test_probs[name] for name in models) / total_weight
    soft_pred = classes[np.argmax(soft_probs, axis=1)]

    st.subheader("Model Weights (Validation Accuracy)")
    st.dataframe(pd.DataFrame({"Model": list(weights.keys()), "Weight": [round(w, 3) for w in weights.values()]}), hide_index=True)
    st.subheader("Voting Results")
    results = pd.DataFrame({"Method": ["Hard Voting", "Soft Voting"], "Test Accuracy": [accuracy_score(y_test, hard_pred), accuracy_score(y_test, soft_pred)]})
    results["Test Accuracy"] = results["Test Accuracy"].round(3)
    st.dataframe(results, hide_index=True)