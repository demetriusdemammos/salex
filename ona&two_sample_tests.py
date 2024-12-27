import numpy as np
import pandas as pd
from scipy import stats
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols

def one_sample_t_test_from_csv(filepath, column, population_mean):
    """
    Performs a one-sample t-test using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column (str): The name of the column containing sample data.
    population_mean (float): The known population mean to compare against.

    Returns:
    t_stat (float): The t-statistic.
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    sample_data = data[column].dropna()  # Drop NaN values

    if sample_data.empty:
        raise ValueError("Sample data is empty after removing NaN values.")

    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)  # Use ddof=1 for sample standard deviation
    n = len(sample_data)

    t_stat = (sample_mean - population_mean) / (sample_std / math.sqrt(n))
    p_value = stats.t.sf(np.abs(t_stat), df=n-1) * 2  # Two-tailed test

    return t_stat, p_value

def z_test_one_mean_from_csv(filepath, column, population_mean, population_std):
    """
    Performs a z-test for one mean using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column (str): The name of the column containing sample data.
    population_mean (float): The known population mean to compare against.
    population_std (float): The population standard deviation (known).

    Returns:
    z_stat (float): The z-statistic.
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    sample_data = data[column].dropna()  # Drop NaN values

    if sample_data.empty:
        raise ValueError("Sample data is empty after removing NaN values.")

    sample_mean = np.mean(sample_data)
    n = len(sample_data)

    z_stat = (sample_mean - population_mean) / (population_std / math.sqrt(n))
    p_value = stats.norm.sf(np.abs(z_stat)) * 2  # Two-tailed test

    return z_stat, p_value

def independent_samples_t_test_from_csv(filepath, column1, column2):
    """
    Performs an independent samples t-test using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column1 (str): The name of the first group's data column.
    column2 (str): The name of the second group's data column.

    Returns:
    t_stat (float): The t-statistic.
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column1 not in data.columns or column2 not in data.columns:
        raise ValueError(f"Required columns '{column1}' or '{column2}' not found in the dataset.")

    group1 = data[column1].dropna()  # Drop NaN values
    group2 = data[column2].dropna()  # Drop NaN values

    if group1.empty or group2.empty:
        raise ValueError("One or both groups have no data after removing NaN values.")

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
    return t_stat, p_value

def paired_samples_t_test_from_csv(filepath, column1, column2):
    """
    Performs a paired samples t-test using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column1 (str): The name of the first condition's data column.
    column2 (str): The name of the second condition's data column.

    Returns:
    t_stat (float): The t-statistic.
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column1 not in data.columns or column2 not in data.columns:
        raise ValueError(f"Required columns '{column1}' or '{column2}' not found in the dataset.")

    condition1 = data[column1].dropna()  # Drop NaN values
    condition2 = data[column2].dropna()  # Drop NaN values

    if condition1.empty or condition2.empty:
        raise ValueError("One or both conditions have no data after removing NaN values.")

    t_stat, p_value = stats.ttest_rel(condition1, condition2)
    return t_stat, p_value

# Example usage
if __name__ == "__main__":
    # Replace 'your_file_path.csv' with the actual path to your CSV file
    filepath = 'sample_dataset.csv'

    try:
        # One-sample t-test example
        t_stat, p_value = one_sample_t_test_from_csv(filepath, 'DGirth', 3.0)
        print("One-sample t-test:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}\n")

        # Z-test for one mean example
        z_stat, p_value = z_test_one_mean_from_csv(filepath, 'DGirth', 3.0, 0.5)
        print("Z-test for one mean:")
        print(f"z-statistic: {z_stat:.4f}, p-value: {p_value:.4f}\n")

        # Independent samples t-test example
        t_stat_ind, p_value_ind = independent_samples_t_test_from_csv(filepath, 'DGirth', 'SexQ')
        print("Independent samples t-test:")
        print(f"t-statistic: {t_stat_ind:.4f}, p-value: {p_value_ind:.4f}\n")

        # Paired samples t-test example
        t_stat_paired, p_value_paired = paired_samples_t_test_from_csv(filepath, 'DGirth', 'SexQ')
        print("Paired samples t-test:")
        print(f"t-statistic: {t_stat_paired:.4f}, p-value: {p_value_paired:.4f}\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
