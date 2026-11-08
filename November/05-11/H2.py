import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import io
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")

st.markdown("""
<style>
.main .block-container {
    width: 100%;
    max-width: 1200px;
    padding: 1rem;
}

.stSlider, .stSelectbox {
    width: 100%;
}

@media (max-width: 768px) {
    .main .block-container {
        padding: 0.6rem;
    }

    h1 {
        font-size: 24px !important;
    }

    h2, h3 {
        font-size: 19px !important;
    }

    .stPlotlyChart {
        width: 100% !important;
    }
}

@media (max-width: 480px) {
    .main .block-container {
        padding: 0.4rem;
    }

    .stPlotlyChart {
        width: 100% !important;
    }
}
</style>
""", unsafe_allow_html = True)

file = st.file_uploader("Upload heart.csv")

@st.cache_resource
def train_model(file_data):
    data = pd.read_csv(io.BytesIO(file_data))
    X = data.drop("target", axis=1)
    y = data["target"]
    model = RandomForestClassifier(random_state = 42)
    model.fit(X, y)
    return model, X

if file:
    model, X = train_model(file.getvalue())

    st.subheader("Adjust Patient Features")
    inputs = {}

    for col in X.columns:
        if X[col].nunique() <= 10:
            inputs[col] = st.radio(col, sorted(X[col].dropna().unique().tolist()))

        elif pd.api.types.is_integer_dtype(X[col]):
            inputs[col] = st.slider(col, int(X[col].min()), int(X[col].max()), int(X[col].median()))

        else:
            inputs[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].median()), step = 0.1)

    row = pd.DataFrame([inputs], columns = X.columns)
    probabilities = model.predict_proba(row)[0]
    prediction = model.classes_[np.argmax(probabilities)]

    st.subheader("Prediction Result")
    st.write("Predicted Class:", prediction)
    st.write("Confidence:", round(max(probabilities) * 100, 2), "%")

    chart = pd.DataFrame({"Class": model.classes_.astype(str), "Probability": probabilities})
    fig = px.bar(chart, x = "Class", y = "Probability", color = "Class", title = "Prediction Confidence")
    fig.update_layout(yaxis_range = [0, 1], autosize = True, margin = dict(l = 20, r = 20, t = 50, b = 30))

    st.plotly_chart(fig, use_container_width = True, config = {"responsive": True})