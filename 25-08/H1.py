import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    text = file.read().decode("utf-8")
    lines = text.splitlines()
    columns = lines[0].split(",")
    rows = []

    for line in lines[1:]:
        rows.append(line.split(","))

    numbers = []

    for j in range(len(columns)):
        column = []

        for row in rows:
            try:
                column.append(float(row[j]))
            except:
                pass

        if len(column) > 1:
            numbers.append(column)

    figure, axis = plt.subplots(figsize=(10, 6))

    for i in range(len(numbers)):
        for j in range(len(numbers)):
            x = numbers[i]
            y = numbers[j]
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)
            top = 0
            x_bottom = 0
            y_bottom = 0

            for k in range(len(x)):
                a = x[k] - x_mean
                b = y[k] - y_mean
                top += a * b
                x_bottom += a * a
                y_bottom += b * b

            correlation = top / (x_bottom * y_bottom) ** 0.5

            if correlation > 0:
                color = "lightblue"
            else:
                color = "lightcoral"

            axis.add_patch(Rectangle((j, i), 1, 1, color=color))
            axis.text(j + 0.5, i + 0.5, round(correlation, 2), ha="center", va="center")

    axis.set_xticks([i + 0.5 for i in range(len(numbers))])
    axis.set_xticklabels(columns)
    axis.set_yticks([i + 0.5 for i in range(len(numbers))])
    axis.set_yticklabels(columns)
    axis.set_xlim(0, len(numbers))
    axis.set_ylim(len(numbers), 0)
    axis.set_title("Correlation Heatmap", fontsize=18, pad=12)
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(figure)