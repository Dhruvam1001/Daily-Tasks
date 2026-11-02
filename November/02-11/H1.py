import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]

    results = []
    
    for size in [0.2, 0.3, 0.4]:
        for seed in [1, 10, 42]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = seed)
            
            model = RandomForestClassifier(random_state = 42)
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = seed, stratify = y)
            
            model = RandomForestClassifier(random_state = 42)
            model.fit(X_train, y_train)
            accuracy2 = accuracy_score(y_test, model.predict(X_test))
            
            results.append([size, seed, accuracy, accuracy2])
            
    report = pd.DataFrame(results, columns = ["Test Size", "Seed", "Random Accuracy", "Stratified Accuracy"])

    st.subheader("Accuracy Results")
    st.dataframe(report.round(3), hide_index = True)

    st.subheader("Average Accuracy")
    st.dataframe(report.groupby("Test Size")[["Random Accuracy", "Stratified Accuracy"]].mean().round(3))

    st.subheader("Model Stability")
    st.write("Random Split Variation:", round(report["Random Accuracy"].std(), 3))
    st.write("Stratified Split Variation:", round(report["Stratified Accuracy"].std(), 3))
    st.write("Lower variation means better model stability.")