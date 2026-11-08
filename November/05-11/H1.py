import streamlit as st
import pandas as pd

st.subheader("User Experience Testing")

if "feedback" not in st.session_state:
    st.session_state.feedback = []

with st.form("feedback_form"):
    task = st.selectbox("Tested Section", ["Dataset Upload", "Patient Inputs", "Model Training", "Prediction Results"])
    rating = st.radio("Easy to Use? (Rating)", [1, 2, 3, 4, 5], format_func = lambda x: "⭐" * x)
    problem = st.selectbox("Interaction Problem", ["None", "Confusing Instructions", "Difficult Inputs", "Slow Response", "Unclear Results", "Navigation Problem"])
    comment = st.text_area("User Feedback")
    submit = st.form_submit_button("Submit")

if submit:
    st.session_state.feedback.append([task, rating, problem, comment])
    st.success("Feedback Submitted")

if st.session_state.feedback:
    report = pd.DataFrame(st.session_state.feedback, columns = ["Section", "Rating", "Problem", "Feedback"])
    st.subheader("Usability Findings")
    st.dataframe(report, hide_index = True)

    common = report["Problem"].mode()[0]

    st.write("Average Usability Rating:", round(report["Rating"].mean(), 2))
    st.write("Most Common Pain Point:", common)

    improvements = {
        "None": "Current interaction is working well.",
        "Confusing Instructions": "Provide clearer instructions.",
        "Difficult Inputs": "Simplify input fields and validation.",
        "Slow Response": "Improve response time and loading feedback.",
        "Unclear Results": "Display clearer prediction results.",
        "Navigation Problem": "Simplify navigation and section labels."
    }

    st.subheader("Interface Refinement")
    st.info(improvements[common])