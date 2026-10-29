import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2, random_state = 42, stratify = y)

    model = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
    row = X_test.iloc[[0]]
    base = X_train.mean()
    shap_values = []

    for col in X.columns:
        sample = row.copy()
        sample[col] = base[col]
        shap_values.append(model.predict_proba(row)[0, 1] - model.predict_proba(sample)[0, 1])

    rng = np.random.default_rng(42)
    samples = pd.DataFrame(rng.normal(X_train.mean(), X_train.std().replace(0, 1), size = (300, X.shape[1])), columns = X.columns)
    samples.iloc[0] = row.iloc[0]
    predictions = model.predict_proba(samples)[:, 1]
    distance = np.sqrt(((samples - row.iloc[0]) ** 2).mean(axis = 1))
    weights = np.exp(-distance / (distance.median() + 1e-8))
    lime_model = Ridge().fit(samples, predictions, sample_weight = weights)
    lime_values = lime_model.coef_ * (row.iloc[0] - X_train.mean()).values
    results = pd.DataFrame({"Feature": X.columns, "SHAP-style": shap_values, "LIME-style": lime_values})
    correlation = results["SHAP-style"].corr(results["LIME-style"], method = "pearson")
    results["Same Direction"] = (np.sign(results["SHAP-style"]) == np.sign(results["LIME-style"]))

    st.subheader("Explanation Comparison")
    st.dataframe(results.round(3), hide_index = True)
    st.write("Pearson correlation:", round(correlation, 3))

    fig, ax = plt.subplots()

    results.set_index("Feature")[["SHAP-style", "LIME-style"]].plot(kind = "bar", ax = ax)
    ax.set_title("SHAP-style vs LIME-style Contributions")
    ax.set_ylabel("Feature contribution")
    ax.set_xlabel("Feature")

    plt.xticks(rotation = 90)
    st.pyplot(fig)
    plt.close()

    st.subheader("Limitations")
    st.write("""
    - SHAP-style: Replacing one feature at a time ignores feature interactions.
    - LIME-style: Results depend on generated samples and the local surrogate.
    - Numerical features: Different scales can distort distances and contributions.
    - Categorical features: Random normal values are unsuitable; use valid category
      sampling and appropriate encoding.
    - Correlation: High correlation means similar contribution patterns, not proof
      that either explanation is correct.
    """)