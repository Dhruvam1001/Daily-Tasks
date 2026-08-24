import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

file = st.file_uploader("Upload the file.", type=["csv"])

if file is not None:
    data = file.read().decode("utf-8")
    lines = data.splitlines()
    columns = lines[0].split(",")
    rows = []

    for line in lines[1:]:
        rows.append(line.split(","))

    figure, axis = plt.subplots(figsize=(14, 8))

    for i in range(len(rows)):
        for j in range(len(columns)):
            if j >= len(rows[i]) or rows[i][j].strip() == "":
                color = "red"
            else:
                color = "gray"

            square = Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black")

    axis.add_patch(square)
    axis.set_xticks([i + 0.5 for i in range(len(columns))])
    axis.set_xticklabels(columns, ha="right")
    axis.set_yticks([])
    axis.set_xlim(0, len(columns))
    axis.set_ylim(len(rows), 0)
    axis.set_title("Missing Values Heatmap")

    st.pyplot(figure)