import streamlit as st
import pandas as pd
import time
import tracemalloc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth = 5),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = []

    for name, model in models.items():
        tracemalloc.start()
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        memory = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
        start = time.perf_counter()
        pred = model.predict(X_test)
        pred_time = time.perf_counter() - start
        results.append([name, round(accuracy_score(y_test, pred), 3), round(train_time, 4), round(pred_time, 4), round(memory, 1)])

    result = pd.DataFrame(results, columns = ["Model", "Accuracy", "Train Time (s)", "Prediction Time (s)", "Memory (KB)"])
    
    st.dataframe(result, hide_index = True)

    best = result.sort_values(["Accuracy", "Train Time (s)"], ascending = [False, True]).iloc[0]

    st.write("Recommended model:", best["Model"])
    st.write("Reason: Highest accuracy, with training time used as a tie-breaker.")