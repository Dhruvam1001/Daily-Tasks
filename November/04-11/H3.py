import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)
    index = st.selectbox("Select Patient Record", range(len(X_test)))
    row = X_test.iloc[[index]]
    votes = [sum(tree.predict(row)[0] == cls for tree in model.estimators_) for cls in model.classes_]
    probabilities = np.array(votes) / len(model.estimators_)
    prediction = model.classes_[np.argmax(probabilities)]
    confidence = probabilities.max() * 100

    st.subheader("Prediction Result")
    st.write("Prediction:", prediction)
    st.write("Confidence:", round(confidence, 2), "%")

    chart = st.empty()

    for step in range(0, 101, 10):
        fig, ax = plt.subplots()
        
        ax.bar([str(c) for c in model.classes_], probabilities * step / 100)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Target Class")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence Distribution")

        chart.pyplot(fig)
        plt.close(fig)
        time.sleep(0.03)