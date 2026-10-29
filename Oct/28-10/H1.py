import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from math import factorial

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    model = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
    base = X.mean()
    results = []
    n = len(X.columns)
    rng = np.random.default_rng(42)
    orders = 30

    for _, row in X.head(5).iterrows():
        contribution = np.zeros(n)
        for _ in range(orders):
            current = base.copy()
            old = model.predict_proba(current.to_frame().T)[0, 1]
            for i in rng.permutation(n):
                col = X.columns[i]
                current[col] = row[col]
                new = model.predict_proba(current.to_frame().T)[0, 1]
                contribution[i] += (new - old) / orders
                old = new

        results.append(contribution)

    values = np.array(results)
    st.success("Shapley-value estimation completed.")
    st.subheader("Individual Feature Contributions")

    fig, ax = plt.subplots()
    
    ax.barh(X.columns, values[0])
    ax.set_title("Feature Contributions to Predicted Probability")
    ax.set_xlabel("Shapley Value")
    ax.set_ylabel("Features")
    ax.axvline(0, linestyle = "--")
    ax.grid(axis = "x", alpha = 0.3)

    st.pyplot(fig)
    st.subheader("Contribution Variability")
    st.dataframe(pd.DataFrame({"Feature": X.columns, "Mean Absolute Contribution": np.abs(values).mean(axis = 0).round(4)}), hide_index = True)

    fig, ax = plt.subplots()

    ax.boxplot(values, tick_labels = X.columns)
    ax.set_title("Variability of Shapley Values Across 5 Records")
    ax.set_xlabel("Features")
    ax.set_ylabel("Shapley Value")
    ax.axhline(0, linestyle = "--")
    ax.grid(axis = "y", alpha = 0.3)

    plt.xticks(rotation = 90)
    st.pyplot(fig)