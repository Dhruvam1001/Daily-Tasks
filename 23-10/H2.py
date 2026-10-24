import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

file = st.file_uploader("Upload heart.csv", type="csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC())
    }

    results = []
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds[name] = pred

        tp = sum((y_test == 1) & (pred == 1))
        tn = sum((y_test == 0) & (pred == 0))
        fp = sum((y_test == 0) & (pred == 1))
        fn = sum((y_test == 1) & (pred == 0))

        acc = (tp + tn) / len(y_test)
        pre = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        train_acc = np.mean(model.predict(X_train) == y_train)

        results.append([name, acc, pre, rec, train_acc, train_acc - acc])
        result = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "Train Accuracy", "Overfit Gap"])

    st.subheader("Performance Metrics")
    st.dataframe(result.round(3), hide_index=True)

    dt, svm = preds["Decision Tree"], preds["SVM"]
    a = sum((dt == y_test) & (svm != y_test))
    b = sum((dt != y_test) & (svm == y_test))
    p = binomtest(min(a, b), a + b, 0.5).pvalue if a + b else 1

    st.subheader("Statistical Significance")
    st.write("McNemar exact-test p-value:", round(p, 4))
    st.write("Significant difference:", p < 0.05)
    st.subheader("Bias-Variance Analysis")
    
    for _, row in result.iterrows():
        st.write(f"{row['Model']}: Train-test gap = " f"{row['Overfit Gap']:.3f}")
        st.write("A large positive gap suggests possible overfitting. "
                 "Low training and test accuracy may suggest underfitting. "
                 "The gap is only an indicator, not a direct measurement of bias or variance.")

    dt_row = result[result["Model"] == "Decision Tree"].iloc[0]
    svm_row = result[result["Model"] == "SVM"].iloc[0]

    st.subheader("Model Selection")

    if p < 0.05:
        st.write("The models differ significantly on this test set. "
                 "Consider accuracy, precision, recall and overfit gap "
                 "to choose the model best suited to your objective.")
    else:
        st.write("No statistically significant difference was detected. "
                 "Prefer the model that better matches your needs for "
                 "precision, recall, interpretability and simplicity.")

    if svm_row["Recall"] > dt_row["Recall"]:
        st.write("SVM has higher recall, which may help identify more positive cases.")
    elif dt_row["Recall"] > svm_row["Recall"]:
        st.write("Decision Tree has higher recall, which may help identify more positive cases.")
    else:
        st.write("Both models have equal recall.")

    if dt_row["Precision"] > svm_row["Precision"]:
        st.write("Decision Tree has higher precision.")
    elif svm_row["Precision"] > dt_row["Precision"]:
        st.write("SVM has higher precision.")
    else:
        st.write("Both models have equal precision.")