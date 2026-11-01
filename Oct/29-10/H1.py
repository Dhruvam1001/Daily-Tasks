import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    model = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
    global1 = model.feature_importances_
    permutation = permutation_importance(model, X_test, y_test, n_repeats = 5, random_state = 42)
    global2 = np.maximum(permutation.importances_mean, 0)
    row = X_test.iloc[[0]]
    base = X_train.mean()
    original = model.predict_proba(row)[0, 1]
    local = []
    
    for col in X.columns:
        changed = row.copy()
        changed[col] = base[col]
        local.append(original - model.predict_proba(changed)[0, 1])
        
    report = pd.DataFrame({
        "Feature": X.columns,
        "RF Importance": global1,
        "Permutation Importance": global2,
        "Local Contribution": local
    })
    
    report["Global Average"] = report[["RF Importance", "Permutation Importance"]].rank(ascending = False).mean(axis = 1)
    report = report.sort_values("Global Average")
    report.insert(0, "Rank", range(1, len(report) + 1))
    
    st.subheader("Feature Importance Report")
    st.dataframe(report.round(3), hide_index = True)
    
    fig, ax = plt.subplots()
    
    report.sort_values("RF Importance").plot(x = "Feature", y = "RF Importance", kind = "barh", ax = ax, legend = False, title = "Global Feature Importance")
    ax.set_xlabel("Importance")
    
    st.pyplot(fig)
    plt.close()
    
    top = report.iloc[0]["Feature"]
    st.subheader("Technical Report")
    st.write(f"""
    **Objective:** Identify important features for predicting the target.

    **Methods:** Random Forest importance, permutation importance,
    and local feature contributions for one test patient.

    **Feature selection:** The highest-ranked feature is {top}.
    Features ranked highly by both global methods are stronger candidates
    for further evaluation.

    **Global vs local:** Global scores describe overall model behavior.
    Local contributions describe how changing one feature affects this
    patient's predicted probability.

    **Limitations:** Correlated features can distort importance scores.
    Local contributions use one-feature-at-a-time replacement and are
    not exact SHAP values. Importance does not prove causation.
    """)