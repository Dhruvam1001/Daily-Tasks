import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #667eea, #764ba2);
}

.block-container {
    padding: 25px 4%;
    max-width: 1500px;
}

.topbar {
    background: white;
    padding: 20px 30px;
    border-radius: 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.brand {
    color: #667eea;
    font-size: 23px;
    font-weight: bold;
}

.nav {
    color: #555;
    font-size: 14px;
}

.heading {
    color: white;
    margin: 20px 0;
}

.panel {
    background: white;
    color: #174b55;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.stFileUploader {
    background: #f5f7fc;
    padding: 15px;
    border: 2px dashed #7185ed;
    border-radius: 12px;
}

.stButton button {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    width: 100%;
    height: 45px;
    font-weight: bold;
}

.stButton button:hover {
    color: white;
    border: none;
}

.result {
    background: linear-gradient(110deg, #70e1b5, #80d5e8);
    padding: 20px;
    text-align: center;
    border-radius: 12px;
    margin-bottom: 18px;
    color: #174b55;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}

.metric {
    background: #f3f6fb;
    padding: 20px 10px;
    text-align: center;
    border-radius: 10px;
}

.metric h2 {
    color: #667eea;
    margin: 0;
}

.metric p {
    color: #777;
    margin: 5px 0 0;
    font-size: 13px;
}

.steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.step {
    background: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: #555;
}

@media(max-width: 600px) {
    .topbar {
        padding: 15px;
        flex-direction: column;
        gap: 10px;
    }

    .brand {
        font-size: 19px;
    }

    .panel {
        padding: 15px;
    }

    .metric {
        padding: 15px 5px;
    }
}
</style>

<div class="topbar">
    <div class="brand">Heart Disease Prediction</div>
    <div class="nav">Dashboard　 Model Results　 Robustness</div>
</div>

<div class="heading">
    <h2>Heart Disease Analysis</h2>
    <p>Machine Learning Based Prediction and Performance Analysis</p>
</div>

<div class="steps">
    <div class="step">01<br><b>Dataset Upload</b></div>
    <div class="step">02<br><b>Preprocessing</b></div>
    <div class="step">03<br><b>Model Training</b></div>
    <div class="step">04<br><b>Evaluation</b></div>
</div>

<div class="panel">
    <h3>Dataset & Model Analysis</h3>
    <p>Upload your heart disease dataset and select a machine learning model.</p>
</div>
""", unsafe_allow_html = True)

file = st.file_uploader("Upload heart.csv")

model = st.selectbox("Select Machine Learning Model", ["Random Forest", "Decision Tree", "KNN"])

if st.button("Train & Analyze Model"):
    if file is None:
        st.error("Please upload heart.csv")

    else:
        data = pd.read_csv(file).dropna()

        if "target" not in data.columns:
            st.error("Dataset must contain a target column.")

        else:
            X = data.drop("target", axis=1)
            y = data["target"]

            if y.nunique() != 2 or y.value_counts().min() < 2:
                st.error("Dataset must contain two target classes.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

                if model == "Random Forest":
                    model = RandomForestClassifier(random_state = 42)
                elif model == "Decision Tree":
                    model = DecisionTreeClassifier(random_state = 42)
                else:
                    model = make_pipeline(StandardScaler(), KNeighborsClassifier())

                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                prob = model.predict_proba(X_test)

                accuracy = accuracy_score(y_test, prediction)
                precision = precision_score(y_test, prediction, zero_division = 0)
                recall = recall_score(y_test, prediction, zero_division = 0)
                f1 = f1_score(y_test, prediction, zero_division = 0)

                sample_prediction = prediction[0]
                confidence = prob[0].max() * 100

                result = ("Heart Disease Detected" if sample_prediction == 1 else "No Heart Disease Detected")

                st.markdown(f"""
                <div class="panel">
                    <h3>Model Results</h3>
                    <div class="result">
                        <h2>{result}</h2>
                        <p>Prediction Confidence: {confidence:.1f}%</p>
                    </div>
                    <div class="metrics">
                        <div class="metric">
                            <h2>{len(data)}</h2>
                            <p>Total Records</p>
                        </div>
                        <div class="metric">
                            <h2>{len(X.columns)}</h2>
                            <p>Total Features</p>
                        </div>
                        <div class="metric">
                            <h2>{accuracy:.1%}</h2>
                            <p>Accuracy</p>
                        </div>
                        <div class="metric">
                            <h2>{f1:.3f}</h2>
                            <p>F1 Score</p>
                        </div>
                        <div class="metric">
                            <h2>{precision:.3f}</h2>
                            <p>Precision</p>
                        </div>
                        <div class="metric">
                            <h2>{recall:.3f}</h2>
                            <p>Recall</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html = True)