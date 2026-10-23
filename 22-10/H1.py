import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "Neural Network": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42))
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        pred = model.predict(X_test)
        
        tp = sum((y_test == 1) & (pred == 1))
        tn = sum((y_test == 0) & (pred == 0))
        fp = sum((y_test == 0) & (pred == 1))
        fn = sum((y_test == 1) & (pred == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        train_acc = sum(y_train == train_pred) / len(y_train)
        test_acc = accuracy
        gap = train_acc - test_acc
        
        results.append([
            name, round(accuracy, 3), round(precision, 3),
            round(recall, 3), round(train_acc, 3),
            round(test_acc, 3), round(gap, 3)
        ])
        
    st.subheader("Performance Comparison")
    st.dataframe(pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "Train Accuracy", "Test Accuracy", "Overfit Gap"]), hide_index=True)
    
    st.subheader("Overfitting Analysis")
    st.write("- A large positive overfit gap may indicate overfitting.")
    st.write("- A small gap suggests more similar training and test performance.")
    
    st.subheader("Deep Learning: Advantages")
    st.write("- Can learn complex patterns from data.")
    st.write("- Can learn useful feature combinations automatically.")
    
    st.subheader("Deep Learning: Limitations")
    st.write("- Often needs more data and computing power.")
    st.write("- Can take longer to train and be harder to explain.")
    st.write("- May overfit small datasets.")
    
    st.subheader("Classical ML: Advantages")
    st.write("- Usually faster and easier to train on small datasets.")
    st.write("- Decision Trees are relatively easy to interpret.")
    
    st.subheader("Classical ML: Limitations")
    st.write("- May struggle with very complex patterns.")
    st.write("- Some models need careful feature preparation and scaling.")