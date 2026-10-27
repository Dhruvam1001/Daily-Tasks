import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

file = st.file_uploader("Upload heart.csv", type="csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    tried = []
    scores = []

    for depth in [2, 5, 8]:
        model = DecisionTreeClassifier(max_depth = depth, random_state = 42)
        model.fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val), zero_division = 0)
        tried.append(depth)
        scores.append(score)

    for _ in range(5):
        gp = GaussianProcessRegressor(kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level = 0.01), normalize_y = True, random_state = 42, n_restarts_optimizer = 2)
        gp.fit(np.array(tried).reshape(-1, 1), np.array(scores))
        candidates = [d for d in range(1, 11) if d not in tried]

        if not candidates:
            break
        mean, std = gp.predict(np.array(candidates).reshape(-1, 1), return_std = True)
        improvement = mean - max(scores)
        z = np.zeros_like(std)
        valid = std > 1e-9
        z[valid] = improvement[valid] / std[valid]
        ei = np.zeros_like(std)
        ei[valid] = (improvement[valid] * norm.cdf(z[valid]) + std[valid] * norm.pdf(z[valid]))
        ei[~valid] = np.maximum(improvement[~valid], 0)
        depth = candidates[int(np.argmax(ei))]
        model = DecisionTreeClassifier(max_depth = depth, random_state = 42)
        model.fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val), zero_division = 0)
        tried.append(depth)
        scores.append(score)

    results = pd.DataFrame({"Max Depth": tried, "F1-score": np.round(scores, 3)})
    best = int(np.argmax(scores))

    st.subheader("Bayesian Optimization Results")
    st.dataframe(results, hide_index=True)
    st.write("Best max_depth:", tried[best])

    best_model = DecisionTreeClassifier(max_depth = tried[best], random_state = 42)
    best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    st.success("Best model trained successfully.")