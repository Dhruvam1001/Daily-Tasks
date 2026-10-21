
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

file = st.file_uploader("Upload heart.csv", type="csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    models = {
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        tp = sum((y_test == 1) & (pred == 1))
        tn = sum((y_test == 0) & (pred == 0))
        fp = sum((y_test == 0) & (pred == 1))
        fn = sum((y_test == 1) & (pred == 0))

        results[name] = [
            (tp + tn) / len(y_test),
            tp / (tp + fp) if tp + fp else 0,
            tp / (tp + fn) if tp + fn else 0
        ]

    scores = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall"]).T

    st.subheader("Performance Metrics (%)")
    st.dataframe((scores * 100).round(2))

    fig, ax = plt.subplots()
    scores.plot(kind="bar", ylim=(0, 1), ax=ax)
    ax.set_title("Model Performance")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Decision Boundaries")
    features = ["age", "thalach"]

    if all(c in X.columns for c in features):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, (name, model) in zip(axes, models.items()):
            model.fit(X_train[features], y_train)

            xx, yy = np.meshgrid(
                np.linspace(X[features[0]].min(), X[features[0]].max(), 40),
                np.linspace(X[features[1]].min(), X[features[1]].max(), 40))
            grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=features)
            zz = model.predict(grid).reshape(xx.shape)

            ax.contourf(xx, yy, zz, alpha=0.3)
            ax.scatter(X_test[features[0]], X_test[features[1]], c=y_test, s=10)
            ax.set_title(name)
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Dataset must contain age and thalach columns.")