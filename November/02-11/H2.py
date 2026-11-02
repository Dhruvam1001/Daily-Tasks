import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    noise = np.random.default_rng(42).normal(0, 0.1, size = X_test.shape)
    perturbed = X_test + noise
    
    worst = X_test.copy()
    worst_acc = accuracy

    for amount in [0.1, 0.2, 0.3, 0.5]:
        changed = X_test + amount
        acc = accuracy_score(y_test, model.predict(changed))

        if acc < worst_acc:
            worst_acc = acc
            worst = changed

    perturbed_acc = accuracy_score(
        y_test, model.predict(perturbed))

    st.subheader("Robustness Results")
    st.write("Original Accuracy:", round(accuracy, 3))
    st.write("Perturbed Accuracy:", round(perturbed_acc, 3))
    st.write("Worst-case Accuracy:", round(worst_acc, 3))
    st.write("Accuracy Drop:", round(accuracy - worst_acc, 3))

    if accuracy - worst_acc > 0.1:
        st.write("Model is sensitive to input changes.")
    else:
        st.write("Model shows better robustness against tested changes.")