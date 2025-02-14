#*****************************
# Name: Jonah Zembower
# Date: February 14, 2025
# Project: Attempting to perform SPM techniques based on the current research
#*****************************

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Function for moving average
def moving_average(data, window_size=200):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function for paired t-test visualization
def spm_paired_ttest(data, col_left, col_right, alpha=0.05, window_size=200):
    left = data[col_left].values
    right = data[col_right].values

    # Compute differences and apply moving average filter
    differences = left - right
    smoothed_diff = moving_average(differences, window_size)

    # Perform paired t-test
    t_values, p_values = stats.ttest_rel(left, right)

    # Create x-axis
    x = np.arange(len(smoothed_diff))

    # Compute confidence intervals
    mean_diff = np.mean(smoothed_diff)
    std_diff = np.std(smoothed_diff)
    ci_upper = mean_diff + 1.96 * (std_diff / np.sqrt(len(smoothed_diff)))
    ci_lower = mean_diff - 1.96 * (std_diff / np.sqrt(len(smoothed_diff)))

    # Dynamic threshold
    dynamic_threshold = np.percentile(np.abs(smoothed_diff), 95)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, smoothed_diff, label=f'Smoothed Difference: {col_left} - {col_right}', color='black')
    ax.axhline(y=dynamic_threshold, color='red', linestyle='--', label='Dynamic Threshold')
    ax.axhline(y=-dynamic_threshold, color='red', linestyle='--')
    ax.fill_between(x, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% Confidence Interval')

    # Highlight significant regions
    significant = np.abs(smoothed_diff) > dynamic_threshold
    ax.scatter(x[significant], smoothed_diff[significant], color='red', label='Significant Deviations')

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Smoothed Difference')
    ax.set_title(f'Paired t-test: {col_left} vs {col_right}')
    ax.legend()
    ax.grid()

    # Show plot
    st.pyplot(fig)

    # Display results
    st.write(f"**T-test result:** t = {t_values:.4f}, p = {p_values:.4f}")
    if p_values < alpha:
        st.success(f"Significant difference found (p < {alpha})")
    else:
        st.warning(f"No significant difference found (p >= {alpha})")


# Streamlit UI
st.title("SPM-Like Paired T-Test Visualization")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # Column Selection
    col_options = df.columns.tolist()
    col_left = st.selectbox("Select first column (left)", col_options)
    col_right = st.selectbox("Select second column (right)", col_options)

    # Parameters
    alpha = st.slider("Select significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
    window_size = st.slider("Select moving average window size", 10, 500, 200, 10)

    # Run Analysis
    if st.button("Run Analysis"):
        spm_paired_ttest(df, col_left, col_right, alpha, window_size)
