#*****************************
# Name: Jonah Zembower
# Date: February 14, 2025
# Project: Attempting to perform SPM techniques based on the current research
#*****************************

# Importing Necessary Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Function for moving average
def moving_average(data, window_size=200):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function for paired t-test visualization
def spm_paired_ttest(data1, data2, col_left, col_right, alpha=0.05, window_size=200):
    left = data1[col_left].values
    right = data2[col_right].values

    # Compute differences and apply moving average filter
    differences = left - right
    smoothed_diff = moving_average(differences, window_size)

    # Create x-axis
    x = np.arange(len(smoothed_diff))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, smoothed_diff, label=f'Smoothed Difference: {col_left} - {col_right}', color='black')

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Smoothed Difference')
    ax.set_title(f'Moving Average of Differences: {col_left} vs {col_right}')
    ax.legend()
    ax.grid()

    # Show plot
    st.pyplot(fig)

# Streamlit UI
st.title("SPM Visualization")

# File Upload
uploaded_file1 = st.file_uploader("Upload your first dataset (CSV or Excel)", type=["csv", "xlsx"], key="file1")
uploaded_file2 = st.file_uploader("Upload your second dataset (CSV or Excel)", type=["csv", "xlsx"], key="file2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    # Read the datasets
    file_extension1 = uploaded_file1.name.split('.')[-1]
    file_extension2 = uploaded_file2.name.split('.')[-1]
    if file_extension1 == 'csv':
        df1 = pd.read_csv(uploaded_file1)
    else:
        df1 = pd.read_excel(uploaded_file1)
    
    if file_extension2 == 'csv':
        df2 = pd.read_csv(uploaded_file2)
    else:
        df2 = pd.read_excel(uploaded_file2)

    st.write("### Preview of First Dataset")
    st.dataframe(df1.head())
    
    st.write("### Preview of Second Dataset")
    st.dataframe(df2.head())

    # Column Selection
    col_options1 = df1.columns.tolist()
    col_options2 = df2.columns.tolist()
    col_left = st.selectbox("Select first column from first dataset", col_options1)
    col_right = st.selectbox("Select second column from second dataset", col_options2)

    # Parameters
    alpha = st.slider("Select significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
    window_size = st.slider("Select moving average window size", 10, 500, 200, 10)

    # Run Analysis
    if st.button("Run Analysis"):
        spm_paired_ttest(df1, df2, col_left, col_right, alpha, window_size)

