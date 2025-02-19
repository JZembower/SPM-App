#*****************************
# Name: Jonah Zembower
# Date: February 14, 2025
# Project: Attempting to perform SPM techniques based on the current research
#*****************************

# Importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# Function for moving average
def moving_average(data, window_size=200):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function for single t-test with visualization
def single_ttest(data, col, alpha=0.05):
    values = data[col].dropna().values
    t_stat, p_value = stats.ttest_1samp(values, 0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(values), color='red', linestyle='dashed', linewidth=2, label='Mean')
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"**Single T-test for {col}:** t = {t_stat:.4f}, p = {p_value:.4f}")
    if p_value < alpha:
        st.success(f"Significant difference found (p < {alpha})")
    else:
        st.warning(f"No significant difference found (p >= {alpha})")

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

# Function for ANOVA with visualization
def anova_test(data, columns, alpha=0.05):
    values = [data[col].dropna().values for col in columns]
    f_stat, p_value = stats.f_oneway(*values)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, labels=columns)
    ax.set_title("ANOVA Boxplot")
    ax.set_ylabel("Values")
    st.pyplot(fig)
    
    st.write(f"**ANOVA Test for {', '.join(columns)}:** F = {f_stat:.4f}, p = {p_value:.4f}")
    if p_value < alpha:
        st.success(f"Significant differences found among the groups (p < {alpha})")
    else:
        st.warning(f"No significant differences found (p >= {alpha})")

# Streamlit UI
st.title("SPM-Like Statistical Tests")

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

    # Select analysis type
    test_type = st.selectbox("Select Statistical Test", ["Single T-Test", "Paired T-Test", "ANOVA"])

    if test_type == "Single T-Test":
        col = st.selectbox("Select column for Single T-Test", df.columns.tolist())
        alpha = st.slider("Select significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
        if st.button("Run Analysis"):
            single_ttest(df, col, alpha)
    
    elif test_type == "Paired T-Test":
        col_options = df.columns.tolist()
        col_left = st.selectbox("Select first column (left)", col_options)
        col_right = st.selectbox("Select second column (right)", col_options)
        alpha = st.slider("Select significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
        window_size = st.slider("Select moving average window size", 10, 500, 200, 10)
        if st.button("Run Analysis"):
            spm_paired_ttest(df, col_left, col_right, alpha, window_size)
    
    elif test_type == "ANOVA":
        columns = st.multiselect("Select multiple columns for ANOVA", df.columns.tolist())
        alpha = st.slider("Select significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
        if len(columns) > 1 and st.button("Run Analysis"):
            anova_test(df, columns, alpha)
        elif len(columns) <= 1:
            st.warning("Please select at least two columns for ANOVA.")
