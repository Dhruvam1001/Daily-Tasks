import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    model = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
    row = X_test.iloc[[0]]
    rng = np.random.default_rng(42)
    scale = X_train.std().replace(0, 1)
    samples = pd.DataFrame(rng.normal(loc = row.iloc[0].values, scale = scale.values, size = (200, X.shape[1])), columns = X.columns)
    samples.iloc[0] = row.iloc[0]
    predictions = model.predict_proba(samples)[:, 1]
    distance = np.sqrt(((samples - row.iloc[0]) ** 2 / (scale ** 2)).mean(axis = 1))
    weights = np.exp(-distance / (distance.median() + 1e-8))
    surrogate = Ridge().fit(samples, predictions, sample_weight = weights)
    check = pd.DataFrame(rng.normal(loc = row.iloc[0].values, scale = scale.values, size = (50, X.shape[1])), columns = X.columns)
    actual = model.predict_proba(check)[:, 1]
    explained = surrogate.predict(check)
    
    st.write("Original prediction:", round(model.predict_proba(row)[0, 1], 3))
    st.write("Independent verification R²:", round(r2_score(actual, explained), 3))

    importance = pd.Series(surrogate.coef_, index = X.columns).sort_values()
    importance.plot(kind = "barh", title = "LIME Feature Contributions")
    
    st.pyplot(plt.gcf())
    plt.close()