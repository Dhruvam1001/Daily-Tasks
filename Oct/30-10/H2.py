import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

file = st.file_uploader("Upload heart.csv", type="csv")

if file:
    data = pd.read_csv(file)
    group_col = "sex"
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    result = X_test[[group_col]].copy()
    result["Actual"] = y_test
    result["Predicted"] = pred
    scores = []

    for group in result[group_col].unique():
        d = result[result[group_col] == group]
        positive_rate = (d["Predicted"] == 1).mean()
        tpr = ((d["Predicted"] == 1) & (d["Actual"] == 1)).sum() / max((d["Actual"] == 1).sum(), 1)
        fpr = ((d["Predicted"] == 1) & (d["Actual"] == 0)).sum() / max((d["Actual"] == 0).sum(), 1)
        scores.append([group, positive_rate, tpr, fpr])

    report = pd.DataFrame(scores, columns = ["Group", "Positive Rate", "TPR", "FPR"])

    st.subheader("Fairness Metrics by Group")
    st.dataframe(report.round(3), hide_index = True)

    impact = (report["Positive Rate"].min() / max(report["Positive Rate"].max(), 1e-8))
    tpr_diff = report["TPR"].max() - report["TPR"].min()
    fpr_diff = report["FPR"].max() - report["FPR"].min()

    st.subheader("Bias Scores")
    st.write("Disparate Impact:", round(impact, 3))
    st.write("TPR Difference:", round(tpr_diff, 3))
    st.write("FPR Difference:", round(fpr_diff, 3))