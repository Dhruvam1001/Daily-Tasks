import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    results = {}
    
    def get_folds(k, stratified):
        folds = [[] for _ in range(k)]
        rng = np.random.default_rng(42)

        if stratified:
            for c in y.unique():
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                for i, part in enumerate(np.array_split(idx, k)):
                    folds[i] += list(part)
        else:
            idx = rng.permutation(len(y))
            folds = [list(p) for p in np.array_split(idx, k)]

        return [np.array(f) for f in folds]

    for k in [3, 5, 10]:
        for method, strat in [("K-Fold", False), ("Stratified K-Fold", True)]:
            folds = get_folds(k, strat)
            scores = []

            for i in range(k):
                test = folds[i]
                train = np.concatenate([folds[j] for j in range(k) if j != i])
                model = RandomForestClassifier(random_state = 42)
                model.fit(X.iloc[train], y.iloc[train])
                scores.append(accuracy_score(y.iloc[test], model.predict(X.iloc[test])))

            results[(k, method)] = scores

            if k == 5 and strat:
                strat_folds = folds

    st.subheader("1. K-Fold and Stratified K-Fold Results")
    st.dataframe(pd.DataFrame([[m, i + 1, round(s, 2)]
        for m in ["K-Fold", "Stratified K-Fold"]
        for i, s in enumerate(results[(5, m)])], columns = ["Method", "Fold", "Model Performance"]), hide_index = True)

    st.subheader("2. Balanced Class Representation")
    st.dataframe(pd.DataFrame([{"Fold": i + 1, **{f"Class {c}": f"{(y.iloc[f] == c).sum()}" f"({(y.iloc[f] == c).mean()*100:.1f}%)"
        for c in y.unique()}} for i, f in enumerate(strat_folds)]), hide_index = True)

    st.subheader("3. Performance Variance Analysis")
    st.dataframe(pd.DataFrame([[m, np.mean(s), np.var(s), np.std(s)]
        for m in ["K-Fold", "Stratified K-Fold"]
        for s in [results[(5, m)]]], columns = ["Method", "Mean Performance", "Variance", "Standard Deviation"]).round(4), hide_index = True)

    st.subheader("4. Performance Across Different Folds")
    st.dataframe(pd.DataFrame([[f"{k}-Fold", m, np.mean(results[(k, m)])]
        for k in [3, 5, 10]
        for m in ["K-Fold", "Stratified K-Fold"]], columns = ["Fold Configuration", "Method", "Mean Performance"]).round(3), hide_index = True)