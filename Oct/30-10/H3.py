import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file = st.file_uploader("Upload heart.csv", type="csv")

if file:
    data = pd.read_csv(file)
    group_col = "sex"
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    groups_train = X_train[group_col]
    groups_test = X_test[group_col]
    weights = groups_train.map(1 / groups_train.value_counts(normalize = True))
    models = {
        "Original": (X_train, X_test, None),
        "Reweighting": (X_train, X_test, weights),
        "Remove Demographic": (X_train.drop(columns = group_col), X_test.drop(columns = group_col), None)
    }

    results = []

    for name, (train_x, test_x, sample_weights) in models.items():
        model = RandomForestClassifier(random_state = 42)
        model.fit(train_x, y_train, sample_weight = sample_weights)
        prediction = model.predict(test_x)
        rates = []
        
        for group in groups_test.unique():
            mask = groups_test == group
            actual = y_test[mask]
            predicted = pd.Series(prediction, index = y_test.index)[mask]
            positive_rate = (predicted == 1).mean()
            tpr = ((predicted == 1) & (actual == 1)).sum() / max((actual == 1).sum(), 1)
            fpr = ((predicted == 1) & (actual == 0)).sum() / max((actual == 0).sum(), 1)
            rates.append([positive_rate, tpr, fpr])

        rates = pd.DataFrame(rates, columns = ["Positive", "TPR", "FPR"])
        impact = rates["Positive"].min() / max(rates["Positive"].max(), 1e-8)
        equalized_odds = max(rates["TPR"].max() - rates["TPR"].min(), rates["FPR"].max() - rates["FPR"].min())
        results.append([name, accuracy_score(y_test, prediction), impact, equalized_odds])

    report = pd.DataFrame(results, columns = ["Strategy", "Accuracy", "Disparate Impact", "Equalized Odds Difference"])

    st.subheader("Bias Mitigation Comparison")
    st.dataframe(report.round(3), hide_index = True)
    st.write("""
    **Interpretation:**
    - Higher accuracy indicates better prediction performance.
    - Disparate Impact closer to 1 indicates more similar positive rates.
    - Equalized Odds Difference closer to 0 indicates better fairness.

    **Conclusion:** Compare all three strategies to identify whether
    fairness improves without an unacceptable loss in accuracy.
    """)