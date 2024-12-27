import numpy as np
import pandas as pd
from scipy.stats import kruskal, friedmanchisquare, spearmanr, kendalltau

def kruskal_wallis_h_test_from_csv(filepath, group_column, value_column):
    """
    Performs the Kruskal-Wallis H Test using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    group_column (str): The column indicating group membership.
    value_column (str): The column containing the values to compare.

    Returns:
    stat (float): The test statistic.
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if group_column not in data.columns or value_column not in data.columns:
        raise ValueError(f"Required columns '{group_column}' or '{value_column}' not found in the dataset.")

    groups = [group[value_column].dropna() for _, group in data.groupby(group_column)]

    if any(len(group) == 0 for group in groups):
        raise ValueError("One or more groups have no data after removing NaN values.")

    stat, p_value = kruskal(*groups)
    return stat, p_value

def friedman_test_from_csv(filepath, subject_column, condition_columns):
    """
    Performs the Friedman Test using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    subject_column (str): The column indicating subject identifiers.
    condition_columns (list of str): The columns containing the values for each condition.

    Returns:
    stat (float): The test statistic.
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if subject_column not in data.columns or not all(col in data.columns for col in condition_columns):
        raise ValueError(f"Required columns '{subject_column}' or '{condition_columns}' not found in the dataset.")

    data = data.dropna(subset=condition_columns)

    if data.empty:
        raise ValueError("No complete cases available for the conditions after removing NaN values.")

    stat, p_value = friedmanchisquare(*(data[col] for col in condition_columns))
    return stat, p_value

def spearman_rank_correlation_from_csv(filepath, column1, column2):
    """
    Calculates Spearman's Rank Correlation between two variables using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column1 (str): The first variable.
    column2 (str): The second variable.

    Returns:
    corr (float): The Spearman correlation coefficient.
    p_value (float): The p-value for the correlation.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column1 not in data.columns or column2 not in data.columns:
        raise ValueError(f"Required columns '{column1}' or '{column2}' not found in the dataset.")

    x = data[column1].dropna()
    y = data[column2].dropna()

    if x.empty or y.empty:
        raise ValueError("One or both columns have no data after removing NaN values.")

    corr, p_value = spearmanr(x, y)
    return corr, p_value

def kendall_tau_from_csv(filepath, column1, column2):
    """
    Calculates Kendall's Tau correlation between two variables using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column1 (str): The first variable.
    column2 (str): The second variable.

    Returns:
    corr (float): The Kendall correlation coefficient.
    p_value (float): The p-value for the correlation.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column1 not in data.columns or column2 not in data.columns:
        raise ValueError(f"Required columns '{column1}' or '{column2}' not found in the dataset.")

    x = data[column1].dropna()
    y = data[column2].dropna()

    if x.empty or y.empty:
        raise ValueError("One or both columns have no data after removing NaN values.")

    corr, p_value = kendalltau(x, y)
    return corr, p_value

# Example usage
if __name__ == "__main__":
    # Replace 'your_file_path.csv' with the actual path to your CSV file
    filepath = 'sample_dataset.csv'

    try:
        # Kruskal-Wallis H Test example
        stat_kw, p_value_kw = kruskal_wallis_h_test_from_csv(filepath, 'DBend', 'SexQ')
        print("Kruskal-Wallis H Test:")
        print(f"stat: {stat_kw:.4f}, p-value: {p_value_kw:.4f}\n")

        # Friedman Test example
        stat_friedman, p_value_friedman = friedman_test_from_csv(filepath, 'num', ['DGirth', 'DLength', 'SexQ'])
        print("Friedman Test:")
        print(f"stat: {stat_friedman:.4f}, p-value: {p_value_friedman:.4f}\n")

        # Spearman's Rank Correlation example
        corr_spearman, p_value_spearman = spearman_rank_correlation_from_csv(filepath, 'DGirth', 'SexQ')
        print("Spearman's Rank Correlation:")
        print(f"corr: {corr_spearman:.4f}, p-value: {p_value_spearman:.4f}\n")

        # Kendall's Tau example
        corr_kendall, p_value_kendall = kendall_tau_from_csv(filepath, 'DGirth', 'SexQ')
        print("Kendall's Tau:")
        print(f"corr: {corr_kendall:.4f}, p-value: {p_value_kendall:.4f}\n")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
