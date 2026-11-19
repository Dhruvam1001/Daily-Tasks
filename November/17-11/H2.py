import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv(Path(__file__).with_name("heart.csv"))

if "target" not in data.columns:
    st.error("Target column is missing.")
    st.stop()

X = data.drop("target", axis=1)
y = data["target"]

if y.nunique() != 2 or y.value_counts().min() < 5:
    st.error("Dataset must contain two classes with at least 5 records each.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

models = {
    "Decision Tree": make_pipeline(SimpleImputer(strategy = "median"), DecisionTreeClassifier(random_state = 42)),
    "Random Forest": make_pipeline(SimpleImputer(strategy = "median"), RandomForestClassifier(random_state = 42)),
    "SVM": make_pipeline(SimpleImputer(strategy = "median"), StandardScaler(), SVC()),
    "KNN": make_pipeline(SimpleImputer(strategy = "median"), StandardScaler(), KNeighborsClassifier())
}

results = []
trained = {}

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    pred = model.predict(X_test)
    predict_time = (time.perf_counter() - start) * 1000 / len(X_test)
    results.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, average = "weighted", zero_division = 0),
        recall_score(y_test, pred, average = "weighted", zero_division = 0),
        f1_score(y_test, pred, average = "weighted", zero_division = 0),
        train_time,
        predict_time
    ])

    trained[name] = model

report = pd.DataFrame(results, columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Training Time (ms)", "Prediction Time (ms)"])
best_name = report.loc[report["F1 Score"].idxmax(), "Model"]
best_model = trained[best_name]
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
cv_scores = cross_val_score(best_model, X, y, cv = cv, scoring = "f1_weighted")
baseline = accuracy_score(y_test, best_model.predict(X_test))
columns = [c for c in X.columns if X[c].nunique() > 10]
noise = np.random.default_rng(42).normal(size = (len(X_test), len(columns)))
robustness = []

for level in [0.05, 0.10, 0.20]:
    changed = X_test.copy()

    if columns:
        std = X_train[columns].std().replace(0, 1)
        changed[columns] = (changed[columns] + noise * std.values * level)

    score = accuracy_score(y_test, best_model.predict(changed))
    robustness.append([level, score])

robust_report = pd.DataFrame(robustness, columns = ["Perturbation Level", "Accuracy"])

st.subheader("1. Project Methodologies")
st.write("""
- Data collection, cleaning, missing-value analysis and exploratory analysis.
- Decision Tree, Random Forest, SVM, KNN and neural network experiments.
- Manual hyperparameter tuning, Bayesian optimization and ensemble learning.
- Accuracy, precision, recall, F1-score and cross-validation evaluation.
- Feature importance, SHAP-style and LIME-style explanations.
- Fairness analysis, bias mitigation and misclassification investigation.
- Adversarial testing, OOD testing and model robustness evaluation.
- Streamlit UI development, FastAPI deployment and API performance testing.
""")

st.subheader("2. Experimental Results")
st.write("Dataset Records:", len(data))
st.write("Input Features:", len(X.columns))
st.write("Missing Values:", int(data.isnull().sum().sum()))
st.write("Train-test split: 80% training and 20% testing.")
st.dataframe(report.round(3), hide_index=True)
st.write("Best Model:", best_name)
st.write("5-Fold Mean F1 Score:", round(cv_scores.mean(), 3))
st.write("5-Fold F1 Variation:", round(cv_scores.std(), 3))
st.write("Original Accuracy:", round(baseline, 3))
st.dataframe(robust_report.round(3), hide_index=True)
st.subheader("3. Deployment Challenges")
st.write("""
- Streamlit and FastAPI require separate running services.
- API connection failures occur when the Uvicorn server is unavailable.
- Localhost API URLs must be replaced with deployed cloud URLs.
- Model files and library versions must remain compatible.
- Cloud API authentication and configuration require secure credentials.
- Concurrent requests can increase CPU usage and response time.
- Streamlit Community Cloud has hosting and scaling limitations.
""")

st.subheader("4. Recommendations for Future Improvements")

st.write("""
- Increase dataset size and validate using external datasets.
- Perform repeated cross-validation and hyperparameter optimization.
- Improve model calibration and monitor false negatives.
- Validate feature explanations with domain experts.
- Test fairness across demographic groups.
- Improve caching and monitor API response times.
- Use secure cloud authentication and versioned model files.
- Monitor resource usage and configure autoscaling.
- Implement multi-region deployment and failure monitoring
  for production-level availability.
""")

document = f"""
# FINAL PROJECT IMPLEMENTATION DOCUMENTATION

## 1. Project Methodologies

Data cleaning, exploratory analysis, machine learning,
hyperparameter optimization, ensemble learning,
model interpretability, fairness analysis, robustness testing,
Streamlit development and FastAPI deployment.

## 2. Experimental Results

Dataset Records: {len(data)}
Features: {len(X.columns)}
Missing Values: {data.isnull().sum().sum()}

{report.round(3).to_string(index=False)}

Best Model: {best_name}
Five-Fold Mean F1: {cv_scores.mean():.3f}
Five-Fold Standard Deviation: {cv_scores.std():.3f}
Original Accuracy: {baseline:.3f}

Robustness Results:
{robust_report.round(3).to_string(index=False)}

## 3. Deployment Challenges

- Separate Streamlit and FastAPI services.
- API connection and endpoint configuration.
- Model file and library version compatibility.
- Cloud authentication and resource management.
- Concurrent requests and API latency.
- Hosting and autoscaling limitations.

## 4. Future Recommendations

- Expand and externally validate the dataset.
- Improve cross-validation and model tuning.
- Improve calibration, fairness and interpretability.
- Optimize caching and API response time.
- Secure API credentials and monitor performance.
- Implement multi-region hosting and failover.
"""

st.download_button("Download Final Project Report", document, file_name = "Heart_Disease_Project_Report.txt", mime = "text/plain")