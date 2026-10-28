import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

file = st.file_uploader("Upload heart.csv", type="csv")

if file:
    data = pd.read_csv(file)
    X, y = data.drop("target", axis=1), data["target"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    def f1(y, p):
        tn, fp, fn, tp = confusion_matrix(y, p, labels = [0, 1]).ravel()
        return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0

    configs = {
        "Tree": (DecisionTreeClassifier, "max_depth", [3, 5, 10]),
        "SVM": (SVC, "C", [0.1, 1, 10]),
        "KNN": (KNeighborsClassifier, "n_neighbors", [3, 5, 7])
    }

    results = []
    progress = []

    for name, (cls, param, values) in configs.items():
        def build(v):
            model = cls(**{param: v})
            return make_pipeline(StandardScaler(), model) if name != "Tree" else model

        base = build(values[0])
        base.fit(X_train, y_train)
        before = f1(y_val, base.predict(X_val))
        best = None
        best_score = -1
        best_so_far = -1

        for v in values:
            model = build(v)
            model.fit(X_train, y_train)
            score = f1(y_val, model.predict(X_val))

            best_so_far = max(best_so_far, score)
            progress.append([name, v, score, best_so_far])
            
            if score > best_score:
                best = model
                best_score = score

        train_score = f1(y_train, best.predict(X_train))
        results.append([name, before, best_score, train_score-best_score])

    df = pd.DataFrame(results, columns = ["Model", "Before", "After", "Generalization Gap"]).round(3)

    st.subheader("Before vs After Tuning")
    st.dataframe(df, hide_index = True)
    st.bar_chart(df.set_index("Model")[["Before", "After"]])
    st.subheader("Tuning Progress")
    
    prog = pd.DataFrame(progress, columns = ["Model", "Parameter", "F1-score", "Best So Far"])
    
    fig, ax = plt.subplots()
    
    for name, part in prog.groupby("Model"):
        ax.plot(part["Parameter"], part["Best So Far"], marker = "o", label = name)

    ax.set_ylabel("Best Validation F1-score")
    ax.set_xlabel("Hyperparameter Value")
    ax.legend()
    
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Convergence and Generalization Analysis")

    for _, r in df.iterrows():
        gap = r["Generalization Gap"]
        st.write(f"{r['Model']}: Improvement = {r['After']-r['Before']:.3f}, Gap = {gap:.3f}")
        if gap > 0.10:
            st.write("Possible overfitting.")
        elif r["After"] < 0.60:
            st.write("Possible underfitting or weak performance.")
        else:
            st.write("Check training and validation scores for generalization.")