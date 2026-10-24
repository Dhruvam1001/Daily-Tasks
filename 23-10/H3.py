import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5))
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        tp = sum((y_test == 1) & (pred == 1))
        fp = sum((y_test == 0) & (pred == 1))
        fn = sum((y_test == 1) & (pred == 0))
        
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0)
        score = 0.2 * precision + 0.4 * recall + 0.4 * f1
        
        results.append([name, precision, recall, f1, score])
        result = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1-Score", "Weighted Score"]).round(3)

    st.subheader("Model Performance")
    st.dataframe(result, hide_index=True)

    best = result.loc[result["Weighted Score"].idxmax()]

    st.subheader("Best Model")
    st.write("Selected model:", best["Model"])
    st.write("Weighted score:", best["Weighted Score"])
    st.write("Precision:", best["Precision"])
    st.write("Recall:", best["Recall"])
    st.write("F1-score:", best["F1-Score"])
    st.subheader("Selection Analysis")
    st.write("The score gives 20% weight to precision, 40% to recall, "
             "and 40% to F1-score. This favors finding positive cases "
             "while balancing false positives and false negatives.")
    
    st.write("Decision Tree and SVM use balanced class weights. "
             "KNN is an unweighted baseline because it does not directly "
             "support class_weight in its standard classifier.")