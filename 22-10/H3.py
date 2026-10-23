import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file = st.file_uploader("Upload heart.csv", type="csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = [
        GridSearchCV(make_pipeline(StandardScaler(), SVC(probability=True)), {"svc__C": [1, 10]}, cv=3),
        GridSearchCV(DecisionTreeClassifier(random_state=42), {"max_depth": [3, 5, None]}, cv=3)
    ]

    names = ["SVM", "Decision Tree"]
    train_meta, test_meta, results = [], [], []

    for name, model in zip(names, models):
        train_prob = cross_val_predict(model, X_train, y_train, cv=3, method="predict_proba")[:, 1]
        train_meta.append(train_prob)
        model.fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_meta.append(test_prob)
        results.append([name, model.best_params_, accuracy_score(y_test, test_prob >= 0.5)])

    meta = LogisticRegression()
    meta.fit(pd.DataFrame(train_meta).T, y_train)
    pred = meta.predict(pd.DataFrame(test_meta).T)
    results.append(["Stacking (Logistic Regression)", "-", accuracy_score(y_test, pred)])
    st.subheader("Best Base-Model Parameters")
    
    for name, model in zip(names, models):
        st.write(name, model.best_params_)
        st.subheader("Model Comparison")
        
    result_df = pd.DataFrame(results, columns=["Model", "Best Parameters", "Accuracy"])
    result_df["Accuracy"] = result_df["Accuracy"].round(3)
    
    st.dataframe(result_df, hide_index=True)
    st.write("Meta-model: Logistic Regression")
    st.write("Final Stacking Accuracy:", round(accuracy_score(y_test, pred), 3))