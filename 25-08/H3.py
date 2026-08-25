import streamlit as st
import matplotlib.pyplot as plt

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    lines = file.read().decode("utf-8").splitlines()
    columns = lines[0].split(",")
    rows = []

    for line in lines[1:]:
        rows.append(line.split(","))

    feature = st.selectbox("Select Feature", columns)
    target = st.selectbox("Select Target", columns)
    f = columns.index(feature)
    t = columns.index(target)
    group0 = []
    group1 = []

    for row in rows:
        try:
            value = float(row[f])
            target_value = float(row[t])
            if target_value == 0:
                group0.append(value)
            if target_value == 1:
                group1.append(value)
        except:
            pass

    if len(group0) == 0 or len(group1) == 0:
        st.write("Target must contain 0 and 1 values.")
    else:
        figure, axis = plt.subplots()

        position = 1

        for values in [group0, group1]:
            low = min(values)
            high = max(values)

            if low == high:
                step = 1
            else:
                step = (high - low) / 20
            points = []
            density = []

            for i in range(21):
                point = low + i * step
                total = 0

                for value in values:
                    z = (point - value) / step
                    total = total + (2.718 ** (-z * z / 2))
                points.append(point)
                density.append(total)

            biggest = max(density)
            left = []
            right = []

            for d in density:
                width = d / biggest * 0.4
                left.append(position - width)
                right.append(position + width)

            axis.fill(left + right[::-1], points + points[::-1])

            position = position + 1

        axis.set_xticks([1, 2])
        axis.set_xticklabels(["0", "1"])
        axis.set_xlabel("Target")
        axis.set_ylabel(feature)
        axis.set_title(feature + " vs Target")

        st.pyplot(figure)