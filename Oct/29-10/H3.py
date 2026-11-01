import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X, y = data.drop("target", axis = 1), data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    model = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train)

    model.fit(X_fit, y_fit)
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels = [0, 1]).ravel()
    st.write("True Negatives:", tn, "| False Positives:", fp)
    st.write("False Negatives:", fn, "| True Positives:", tp)
    st.write("Accuracy:", round(accuracy_score(y_test, pred), 3))
    st.write("Brier score (lower is better):", round(brier_score_loss(y_test, prob), 3))

    errors = X_test.copy()
    errors["Actual"] = y_test
    errors["Predicted"] = pred
    errors["Probability"] = prob
    errors["Error Type"] = "Correct"
    errors.loc[(y_test == 0) & (pred == 1), "Error Type"] = "False Positive"
    errors.loc[(y_test == 1) & (pred == 0), "Error Type"] = "False Negative"

    st.subheader("Misclassified Examples")
    st.dataframe(errors[errors["Error Type"] != "Correct"])
    st.subheader("Error Pattern Analysis")
    st.dataframe(errors.groupby("Error Type")[X.columns.tolist()].mean().round(2))

    actual_rate, predicted_rate = calibration_curve(y_test, prob, n_bins = 5, strategy = "uniform")

    fig, ax = plt.subplots()

    ax.plot(predicted_rate, actual_rate, marker = "o", label = "Model")
    ax.plot([0, 1], [0, 1], "--", label = "Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Actual Positive Rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    
    st.pyplot(fig)
    plt.close()