import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from pathlib import Path

data = pd.read_csv(Path(__file__).with_name("heart.csv"))
X = data.drop("target", axis=1)
url = st.text_input("H2 API URL", "http://127.0.0.1:8000")
token = st.text_input("API Token (optional)", type="password")
index = st.selectbox("Select Patient Record", range(len(X)))
count = st.number_input("Number of Requests", 2, 20, 10)

if st.button("Test API Performance"):
    original = {c: float(X.iloc[index][c]) for c in X.columns}
    optimized = {c: float(np.float32(v)) for c, v in original.items()}
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def send_request(payload):
        start = time.perf_counter()
        response = requests.post(url.rstrip("/") + "/predict", json = payload, headers = headers, timeout = 30)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        return response.json(), elapsed

    try:
        normal_result, normal_time = send_request(original)
        results = []

        for i in range(count):
            result, total_time = send_request(optimized)
            results.append([
                i + 1,
                round(total_time, 2),
                result.get("data_processing_ms"),
                result.get("inference_ms"),
                result.get("cache_hit", False)
            ])

        report = pd.DataFrame(results, columns = ["Request", "Response Time (ms)", "Data Processing (ms)", "Inference (ms)", "Cache Hit"])

        st.subheader("Latency Results")
        st.write("Original Response Time:", round(normal_time, 2), "ms")
        st.write("Average Response Time:", round(report["Response Time (ms)"].mean(), 2), "ms")
        st.write("Prediction:", normal_result["prediction"])
        st.dataframe(report, hide_index = True)

        processing = report["Data Processing (ms)"].dropna()
        inference = report["Inference (ms)"].dropna()

        st.subheader("Bottleneck Analysis")

        if len(processing) and len(inference):
            if processing.mean() > inference.mean():
                st.warning("Data processing is the main bottleneck.")
            else:
                st.warning("Model inference is the main bottleneck.")

        else:
            st.info("API does not provide separate processing and inference timings.")

        hits = report["Cache Hit"].sum()
        st.subheader("Caching Performance")
        st.write("Cache Hits:", int(hits))
        st.write("Cache Hit Rate:", round(hits / count * 100, 2), "%")
        st.subheader("Input Precision Optimization")
        fast_result, _ = send_request(optimized)

        st.write("Original and float32 predictions match:", normal_result["prediction"] == fast_result["prediction"])
        st.subheader("Response Time Graph")
        st.bar_chart(report.set_index("Request")["Response Time (ms)"])

    except requests.RequestException as e:
        st.error(f"API connection failed: {e}")