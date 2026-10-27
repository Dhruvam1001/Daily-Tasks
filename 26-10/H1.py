import streamlit as st
import pandas as pd
import joblib
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state = 42),
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier())
    }
    
    scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = f1_score(y_val, model.predict(X_val))
        
    
    best_name = max(scores, key=scores.get)
    best_model = models[best_name]

    X_final = pd.concat([X_train, X_val])
    y_final = pd.concat([y_train, y_val])
    best_model.fit(X_final, y_final)

    artifact = {
        "model": best_model,
        "version": "1.0",
        "features": list(X.columns),
        "sklearn_version": sklearn.__version__
    }

    joblib.dump(artifact, "best_model_v1.joblib", compress=3)
    
    size = os.path.getsize("best_model_v1.joblib")
    saved = joblib.load("best_model_v1.joblib")
    
    st.write("Model version:", artifact["version"])
    st.write("Model storage efficiency:", round(size / 1024, 2), "KB")
    
    if saved["sklearn_version"] != sklearn.__version__:
        st.error("Scikit-learn version mismatch!")
    
    elif saved["features"] != list(X.columns):
        st.error("Feature names or order mismatch!")
    
    else:
        st.success("Model is compatible for deployment.")
 
    with open("best_model_v1.joblib", "rb") as f:
        st.download_button("Download Best Model", data=f, file_name="best_model_v1.joblib")