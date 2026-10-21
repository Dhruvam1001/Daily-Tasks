import streamlit as st
import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from skopt import BayesSearchCV
from skopt.space import Categorical, Real

warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y = make_classification(n_samples = 500, n_features = 10, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

class MLPWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, architecture="8", alpha=0.0001, learning_rate_init=0.001, max_iter=1000, random_state=42):
        self.architecture = architecture
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        layers = tuple(int(n) for n in self.architecture.split(","))
        self.model_ = MLPClassifier(hidden_layer_sizes=layers, alpha=self.alpha, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, random_state=self.random_state)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

if st.button("Tune Model"):
    with st.spinner("Training models and tuning parameters..."):
        results = []
        architectures = ["8", "16", "8,8", "16,16"]

        for architecture in architectures:
            layers = tuple(int(n) for n in architecture.split(","))
            model = MLPClassifier(hidden_layer_sizes=layers, max_iter = 1000, random_state = 42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            results.append({
                "Architecture": str(layers),
                "Test Accuracy": round(accuracy_score(y_test, predictions), 4)
            })

        st.subheader("Architecture Comparison")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        search = BayesSearchCV(
            estimator=MLPWrapper(),
            search_spaces={
                "architecture": Categorical(["8", "16", "8,8", "16,16"]),
                "alpha": Real(1e-5, 1e-1, prior="log-uniform"),
                "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform")
            }, n_iter=10, cv=3, scoring="accuracy", random_state=42, n_jobs=1, error_score="raise")

        search.fit(X_train, y_train)

    st.subheader("Bayesian Optimization Results")

    best_params = search.best_params_
    best_architecture = tuple(int(n) for n in best_params["architecture"].split(","))

    st.write("Best architecture:", best_architecture)
    st.write("Best alpha:", best_params["alpha"])
    st.write("Best learning rate:", best_params["learning_rate_init"])
    st.write("Best cross-validation accuracy:", round(search.best_score_, 4))

    test_predictions = search.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    st.write("Final test accuracy:", round(test_accuracy, 4))
    st.success("Training and tuning completed successfully!")