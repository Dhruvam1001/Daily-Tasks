import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    model = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
    prediction = model.predict(X_test)
    errors = X_test.copy()
    errors["Actual"] = y_test
    errors["Predicted"] = prediction
    errors["Error"] = np.where((y_test == 0) & (prediction == 1), "False Positive", np.where((y_test == 1) & (prediction == 0), "False Negative", "Correct"))

    errors = errors[errors["Error"] != "Correct"].copy()

    if len(errors) >= 2:
        st.subheader("1. Misclassified Records")
        st.dataframe(errors)

        scaled = StandardScaler().fit_transform(errors[X.columns])
        errors["Cluster"] = KMeans(n_clusters = 2, random_state = 42, n_init = 10).fit_predict(scaled)

        st.subheader("2. Error Clusters")
        st.dataframe(errors.groupby("Cluster")[X.columns].mean().round(2))

        if len(errors) >= 5:
            errors["Anomaly"] = LocalOutlierFactor(n_neighbors = min(5, len(errors)-1)).fit_predict(scaled)
            st.subheader("3. Anomaly Detection")
            st.dataframe(errors[errors["Anomaly"] == -1])
        st.subheader("4. Root Cause Analysis")
        st.dataframe(errors.groupby("Error")[X.columns].mean().round(2))
        st.subheader("5. Subgroup Analysis")
        groups = pd.crosstab(errors["Error"], errors["Cluster"])
        st.dataframe(groups)

        total = len(X_test)
        st.write("False positive rate:",
                 round(((y_test == 0) & (prediction == 1)).sum() /
                       max((y_test == 0).sum(), 1), 3))
        st.write("False negative rate:",
                 round(((y_test == 1) & (prediction == 0)).sum() /
                       max((y_test == 1).sum(), 1), 3))

        st.write("""
        **Root-cause findings:** Compare feature averages across error
        types and clusters to identify possible patterns.

        **Limitations:** Clusters are exploratory, not confirmed causes.
        Anomalies are unusual error records, not necessarily incorrect data.
        Subgroup analysis requires sufficient samples in each group.
        """)
    else:
        st.write("Not enough misclassified records for clustering.")