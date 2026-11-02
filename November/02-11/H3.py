import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)
    
    prediction = model.predict(X_test)
    prob = model.predict_proba(X_test).max(axis=1)
    
    st.write("Normal Test Accuracy:", round(accuracy_score(y_test, prediction), 3))
    st.write("Normal Average Confidence:", round(prob.mean(), 3))
    
    features = [c for c in X.columns if X[c].nunique() > 10]
    samples = []
    details = []

    for col in features:
        for label, value in [("Below Minimum", X_train[col].min() - X_train[col].std()), ("Above Maximum", X_train[col].max() + X_train[col].std())]:
            sample = X_test.iloc[0].copy()
            sample[col] = value
            samples.append(sample)
            details.append([col, label])

    if samples:
        ood = pd.DataFrame(samples)
        probabilities = model.predict_proba(ood)
        report = pd.DataFrame(details, columns = ["Feature", "Edge Case"])
        report["Prediction"] = model.predict(ood)
        report["Confidence"] = probabilities.max(axis=1).round(3)

        st.subheader("Synthetic Edge Case Results")
        st.dataframe(report, hide_index = True)
        st.write("OOD Average Confidence:", round(probabilities.max(axis=1).mean(), 3))
        st.write("Low Confidence Cases:", (probabilities.max(axis=1) < 0.6).sum())
    else:
        st.write("No continuous features available for edge testing.")