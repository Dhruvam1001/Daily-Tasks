import streamlit as st
import matplotlib.pyplot as plt

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    data = file.read().decode("utf-8")
    lines = data.splitlines()
    columns = lines[0].split(",")
    rows = []
    
    for line in lines[1:]:
        rows.append(line.split(","))
        
    column = st.selectbox("Slect any column.", columns)
    col_num = columns.index(column)
    values = []
    counts = []
    
    for row in rows[1:]:
        value = row[col_num]
        values.sort()
        if value in values:
            position = values.index(value)
            counts[position] = counts[position] + 1
        else:
            values.append(value)
            counts.append(1)
            
    st.write("Counting..")
    for i in range(len(values)):
        st.write(values[i], ":", counts[i])
    if len(values) <= 5:
        width = 0.6
    else:
        width = 0.4
        
    figure, axis = plt.subplots()
    
    axis.bar(values, counts, width = width)
    axis.set_xlabel(column)
    axis.set_ylabel("Count")
    axis.set_title("Bar Plot of " + column)
    axis.set_xticks(values[::3])
    axis.set_xticklabels(values[::3])
    
    st.pyplot(figure)