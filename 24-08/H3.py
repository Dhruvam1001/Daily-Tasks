import streamlit as st
import matplotlib.pyplot as plt

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    text = file.read().decode("utf-8")
    lines = text.splitlines()
    columns = lines[0].split(",")
    rows = []

    for line in lines[1:]:
        rows.append(line.split(","))

    x_name = st.selectbox("Select X column", columns)
    y_name = st.selectbox("Select Y column", columns)
    x_num = columns.index(x_name)
    y_num = columns.index(y_name)
    x = []
    y = []

    for row in rows:
        if row[x_num] != "" and row[y_num] != "":
            x.append(float(row[x_num]))
            y.append(float(row[y_num]))

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    top = 0
    bottom_x = 0
    bottom_y = 0

    for i in range(len(x)):
        a = x[i] - x_mean
        b = y[i] - y_mean
        top = top + a * b
        bottom_x = bottom_x + a * a
        bottom_y = bottom_y + b * b
    correlation = top / (bottom_x * bottom_y) ** 0.5

    st.write("Correlation:", round(correlation, 2))

    if abs(correlation) >= 0.7:
        st.write("Highly correlated!")
        plt.scatter(x, y)

        slope = top / bottom_x
        intercept = y_mean - slope * x_mean
        line_y = []

        for value in x:
            line_y.append(slope * value + intercept)

        plt.plot(x, line_y)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Scatter Plot with Trendline")
        st.pyplot(plt)

    else:
        st.write("Not highly correlated.")