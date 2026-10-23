import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning

file = st.file_uploader("Upload heart.csv")

if file:
    data = pd.read_csv(file)
    X = data.drop("target", axis = 1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    models = MLPClassifier()    
    train_loss = []
    val_loss = []
    epochs = 50

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        for epoch in range(epochs):
            models.fit(X_train, y_train)
            train_loss.append(models.loss_)
            val_pred = models.predict_proba(X_test)
            val_loss.append(log_loss(y_test, val_pred))

    fig, ax = plt.subplots()
    ax.plot(range(1, epochs + 1), train_loss, label="Training Loss")
    ax.plot(range(1, epochs + 1), val_loss, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.subheader("Learning Trend Analysis")

    if val_loss[-1] < val_loss[0]:
        st.write("Validation loss decreased: the model's validation performance improved.")
    else:
        st.write("Validation loss did not decrease overall.")
    if val_loss[-1] > train_loss[-1]:
        st.write("Validation loss is higher than training loss; overfitting may be occurring.")
    else:
        st.write("Validation loss is not higher than training loss at the final epoch.")