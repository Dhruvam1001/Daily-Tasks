import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
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
    row = X_test.iloc[index]
    baseline = X_train.median().values
    class_index = list(model.classes_).index(1)

    def predict_prob(values):
        sample = pd.DataFrame([values], columns = X.columns)
        return model.predict_proba(sample)[0, class_index]

    base_prob = predict_prob(baseline)
    shap_values = np.zeros(len(X.columns))
    rng = np.random.default_rng(42)

    for _ in range(50):
        current = baseline.copy()
        old_prob = base_prob
        for i in rng.permutation(len(X.columns)):
            current[i] = row.iloc[i]
            new_prob = predict_prob(current)
            shap_values[i] += new_prob - old_prob
            old_prob = new_prob

    shap_values = shap_values / 50
    probabilities = model.predict_proba(row.to_frame().T)[0]
    prediction = model.classes_[np.argmax(probabilities)]

    st.subheader("Prediction Result")
    st.write("Predicted Class:", prediction)
    st.write("Confidence:", round(max(probabilities) * 100, 2), "%")

    report = pd.DataFrame({"Feature": X.columns, "SHAP Value": shap_values}).sort_values("SHAP Value")

    st.subheader("Feature Contributions")
    st.dataframe(report.round(4), hide_index = True)

    fig = px.bar(report, x = "SHAP Value", y = "Feature", orientation = "h", color = "SHAP Value", title = "SHAP Feature Contributions", hover_data = ["SHAP Value"])

    st.plotly_chart(fig, use_container_width = True)
    st.write("Baseline Probability:", round(base_prob, 3))
    st.write("Positive contribution increases predicted probability.")
    st.write("Negative contribution decreases predicted probability.")