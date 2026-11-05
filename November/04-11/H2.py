import streamlit as st
import pandas as pd
import threading
import hashlib
import io
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model(file_data):
    data = pd.read_csv(io.BytesIO(file_data))
    X = data.drop("target", axis=1)
    y = data["target"]
    model = RandomForestClassifier(random_state = 42).fit(X, y)
    return model, X.columns, X.min(), X.max(), X.median()

@st.cache_resource
def get_cache():
    return {}, threading.Lock()

cache, lock = get_cache()

file = st.file_uploader("Upload heart.csv")

if file:
    model, columns, mins, maxs, medians = train_model(file.getvalue())
    inputs = {}

    with st.form("prediction"):
        for col in columns:
            inputs[col] = st.number_input(col, min_value = int(mins[col]), max_value = int(maxs[col]), value = int(medians[col]))

        submit = st.form_submit_button("Predict")

    if submit:
        key = (hashlib.sha256(file.getvalue()).hexdigest(), tuple(inputs.values()))

        with lock:
            result = cache.get(key)

        if result is None:
            row = pd.DataFrame([inputs], columns = columns)
            prediction = int(model.predict(row)[0])
            confidence = float(model.predict_proba(row).max() * 100)
            result = (prediction, confidence)

            with lock:
                if len(cache) >= 500:
                    cache.pop(next(iter(cache)))
                cache[key] = result

        st.subheader("Prediction Result")

        if result[0] == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease Detected")

        st.write("Confidence:", round(result[1], 2), "%")