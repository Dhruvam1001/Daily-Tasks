import streamlit as st
import numpy as np

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int).reshape(-1, 1)
w1 = np.random.randn(4, 8) * 0.1
w2 = np.random.randn(8, 1) * 0.1
gamma, beta = np.ones((1, 8)), np.zeros((1, 8))

if st.button("Train Model"):
    for epoch in range(200):
        z = X @ w1
        mean, var = z.mean(0), z.var(0)
        std = np.sqrt(var + 1e-8)
        n = (z - mean) / std
        a = gamma * n + beta
        h = np.maximum(0, a)
        mask = (np.random.rand(*h.shape) > 0.2) / 0.8
        hd = h * mask
        p = 1 / (1 + np.exp(-np.clip(hd @ w2, -500, 500)))
        d2 = (p - y) / len(X)
        dw2 = hd.T @ d2
        da = (d2 @ w2.T) * mask * (a > 0)
        dg = np.sum(da * n, axis=0, keepdims=True)
        db = np.sum(da, axis=0, keepdims=True)
        dn = da * gamma
        dz = (dn - dn.mean(0) - n * (dn * n).mean(0)) / std
        dw1 = X.T @ dz
        w2 -= 0.01 * dw2
        w1 -= 0.01 * dw1
        gamma -= 0.01 * dg
        beta -= 0.01 * db

        if epoch % 50 == 0:
            loss = -np.mean(y * np.log(p + 1e-8) + (1-y) * np.log(1-p + 1e-8))
            st.write(f"Epoch: {epoch} | Loss: {loss:.4f}")
            st.success("Training completed with Dropout and Batch Normalization!")