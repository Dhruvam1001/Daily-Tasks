import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    @st.cache_resource
    def train_model(X, y):
        return RandomForestClassifier(random_state = 42).fit(X, y)

    model = train_model(X, y)

    st.subheader("Enter Patient Details")
    inputs = {}

    with st.form("prediction"):
        for col in X.columns:
            inputs[col] = st.text_input(col)

        submit = st.form_submit_button("Predict")

    if submit:
        values = {}

        for col, val in inputs.items():
            if not val.strip():
                st.error(col + " is required")
                st.stop()

            try:
                num = float(val)
            except ValueError:
                st.error(col + " must be a number")
                st.stop()

            if not np.isfinite(num):
                st.error(col + " has an invalid value")
                st.stop()

            if num < X[col].min() or num > X[col].max():
                st.error(col + " is out of range")
                st.stop()

            values[col] = num

        result = pd.DataFrame([values], columns = X.columns)
        prediction = model.predict(result)[0]

        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease Detected")