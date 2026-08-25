import streamlit as st
import matplotlib.pyplot as plt

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    lines = file.read().decode("utf-8").splitlines()
    columns = lines[0].split(",")
    data = []

    for line in lines[1:]:
        data.append(line.split(","))

    xc = st.selectbox("X column", columns)
    yc = st.selectbox("Y column", columns)
    ec = st.selectbox("Color/Size column", columns)
    x = []
    y = []
    extra = []

    for row in data:
        try:
            x.append(float(row[columns.index(xc)]))
            y.append(float(row[columns.index(yc)]))
            extra.append(float(row[columns.index(ec)]))
        except:
            pass

    def scatter_plot(x, y, extra):
        sizes = []
        for value in extra:
            sizes.append(value * 2)

        plt.scatter(x, y)
        plt.xlabel(xc)
        plt.ylabel(yc)
        plt.title("Custom Scatter Plot")
        st.pyplot(plt)

    scatter_plot(x, y, extra)