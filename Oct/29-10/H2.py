import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    model = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
    prediction = model.predict(X_test)
    
    st.subheader("1. Model Performance")
    st.write("Accuracy:", round(accuracy_score(y_test, prediction), 3))
    
    importance = permutation_importance(model, X_test, y_test, n_repeats = 5, random_state = 42)
    report = pd.DataFrame({"Feature": X.columns, "Global Importance": importance.importances_mean}).sort_values("Global Importance", ascending = False)
    
    st.subheader("2. Global Feature Importance")
    st.dataframe(report, hide_index = True)
    
    fig, ax = plt.subplots()
    
    report.sort_values("Global Importance").plot(x = "Feature", y = "Global Importance", kind = "barh", legend = False, ax = ax)
    ax.set_title("Global Feature Importance")
    
    st.pyplot(fig)
    plt.close()
    
    row = X_test.iloc[[0]]
    original = model.predict_proba(row)[0, 1]
    local = []
    
    for col in X.columns:
        changed = row.copy()
        changed[col] = X_train[col].mean()
        local.append(
            original - model.predict_proba(changed)[0, 1])

    local_report = pd.DataFrame({"Feature": X.columns, "Local Contribution": local}).sort_values("Local Contribution", key = abs, ascending = False)

    st.subheader("3. Individual Prediction Explanation")
    st.write("Predicted probability of target = 1:", round(original, 3))
    st.dataframe(local_report, hide_index = True)
    st.subheader("4. Domain Expert Feedback")
    feedback = st.text_area("Enter expert observations about the explanations")
    concerns = st.text_area("Enter concerns about clinical usefulness or reliability")

    st.subheader("5. Findings and Improvements")
    st.write("""
    **Findings:** Global importance shows which features influence
    model performance overall. Local contributions show how changing
    one feature affects an individual prediction.

    **Limitations:** Feature importance does not prove causation.
    Correlated features and unrealistic feature changes may mislead.
    Accuracy alone does not measure clinical safety.

    **Proposed improvements:**
    - Validate explanations with domain experts.
    - Evaluate precision, recall, and calibration.
    - Test explanations across different patient groups.
    - Use clinically realistic feature changes.
    - Monitor model performance and explanation stability.
    """)