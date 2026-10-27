import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_score = -1
    best_model = None
    best_params = {}

    for depth in [3, 5, None]:
        for split in [2, 5, 10]:
            model = DecisionTreeClassifier(max_depth=depth, min_samples_split=split, random_state=42)

            model.fit(X_train, y_train)
            score = f1_score(y_val, model.predict(X_val), zero_division=0)

            st.write("Depth:", depth, "Split:", split, "F1-score:", round(score, 3))

            if score > best_score:
                best_score = score
                best_model = model
                best_params = {
                    "max_depth": depth,
                    "min_samples_split": split
                }

    st.subheader("Best Model")
    st.write("Best parameters:", best_params)
    st.write("Best validation F1-score:", round(best_score, 3))