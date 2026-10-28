import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = StandardScaler().fit_transform(data.drop("target", axis=1).to_numpy())
    y = data["target"].to_numpy().reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    np.random.seed(42)
    W1 = np.random.randn(X_train.shape[1], 8) * 0.01
    b1 = np.zeros((1, 8))
    W2 = np.random.randn(8, 1) * 0.01
    b2 = np.zeros((1, 1))
    lr = 0.1
    best = float("inf")
    wait = 0

    for epoch in range(30):
        h = np.tanh(X_train @ W1 + b1)
        p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -500, 500)))
        error = p - y_train
        dW2 = h.T @ error / len(y_train)
        db2 = error.mean()
        dh = (error @ W2.T) * (1 - h**2)
        dW1 = X_train.T @ dh / len(y_train)
        db1 = dh.mean(axis=0, keepdims=True)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        hv = np.tanh(X_val @ W1 + b1)
        pv = 1 / (1 + np.exp(-np.clip(hv @ W2 + b2, -500, 500)))
        loss = -np.mean(y_val * np.log(pv + 1e-8) + (1-y_val) * np.log(1-pv + 1e-8))

        if loss < best:
            best, wait = loss, 0
        else:
            wait += 1

        if wait == 3:
            lr = max(lr / 2, 0.001)
            wait = 0

    st.write("Final learning rate:", lr)
    st.write("Best validation loss:", best)