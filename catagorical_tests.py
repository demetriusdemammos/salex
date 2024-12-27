import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency, fisher_exact, binomtest, norm

def chi_square_goodness_of_fit_test_from_csv(filepath, observed_column, expected_column):
    """
    Performs a Chi-Square Goodness-of-Fit Test using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    observed_column (str): The column containing observed frequencies.
    expected_column (str): The column containing expected frequencies.

    Returns:
    stat (float): The test statistic.
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if observed_column not in data.columns or expected_column not in data.columns:
        raise ValueError(f"Required columns '{observed_column}' or '{expected_column}' not found in the dataset.")

    observed = data[observed_column].dropna()
    expected = data[expected_column].dropna()

    if len(observed) != len(expected):
        raise ValueError("Observed and expected frequencies must have the same length.")

    stat, p_value = chisquare(observed, f_exp=expected)
    return stat, p_value

def chi_square_test_of_independence_from_csv(filepath, column1, column2):
    """
    Performs a Chi-Square Test of Independence using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column1 (str): The first categorical variable.
    column2 (str): The second categorical variable.

    Returns:
    stat (float): The test statistic.
    p_value (float): The p-value of the test.
    dof (int): Degrees of freedom.
    expected (ndarray): The expected frequencies.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column1 not in data.columns or column2 not in data.columns:
        raise ValueError(f"Required columns '{column1}' or '{column2}' not found in the dataset.")

    contingency_table = pd.crosstab(data[column1], data[column2])
    stat, p_value, dof, expected = chi2_contingency(contingency_table)
    return stat, p_value, dof, expected

def fishers_exact_test_from_csv(filepath, column1, column2):
    """
    Performs Fisher's Exact Test using data from a CSV file for 2x2 contingency tables.

    Parameters:
    filepath (str): Path to the CSV file.
    column1 (str): The first categorical variable.
    column2 (str): The second categorical variable.

    Returns:
    odds_ratio (float): The odds ratio.
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column1 not in data.columns or column2 not in data.columns:
        raise ValueError(f"Required columns '{column1}' or '{column2}' not found in the dataset.")

    contingency_table = pd.crosstab(data[column1], data[column2])

    if contingency_table.shape != (2, 2):
        raise ValueError("Fisher's Exact Test requires a 2x2 contingency table.")

    odds_ratio, p_value = fisher_exact(contingency_table)
    return odds_ratio, p_value

def z_test_for_proportions(success_count, total_count, hypothesized_proportion):
    """
    Performs a Z-test for proportions.

    Parameters:
    success_count (int): Number of successes in the sample.
    total_count (int): Total number of observations in the sample.
    hypothesized_proportion (float): The hypothesized population proportion.

    Returns:
    z_stat (float): The Z-statistic.
    p_value (float): The p-value of the test.
    """
    if total_count <= 0:
        raise ValueError("Total count must be greater than 0.")

    sample_proportion = success_count / total_count
    std_error = np.sqrt(hypothesized_proportion * (1 - hypothesized_proportion) / total_count)

    z_stat = (sample_proportion - hypothesized_proportion) / std_error
    p_value = 2 * norm.sf(np.abs(z_stat))  # Two-tailed test
    return z_stat, p_value

def binomial_test_from_csv(filepath, column, hypothesized_proportion):
    """
    Performs a Binomial Test using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column (str): The column containing binary data (1 for success, 0 for failure).
    hypothesized_proportion (float): The hypothesized population proportion.

    Returns:
    p_value (float): The p-value of the test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    sample_data = data[column].dropna()

    if sample_data.empty:
        raise ValueError("Sample data is empty after removing NaN values.")

    success_count = sample_data.sum()
    total_count = len(sample_data)

    p_value = binomtest(success_count, total_count, hypothesized_proportion).pvalue
    return p_value

# Example usage
if __name__ == "__main__":
    # Replace 'your_file_path.csv' with the actual path to your CSV file
    filepath = 'sample_dataset.csv'

    try:
        # Chi-Square Goodness-of-Fit Test example
        stat_chi, p_value_chi = chi_square_goodness_of_fit_test_from_csv(filepath, 'DBend', 'DBend')
        print("Chi-Square Goodness-of-Fit Test:")
        print(f"stat: {stat_chi:.4f}, p-value: {p_value_chi:.4f}\n")

        # Chi-Square Test of Independence example
        stat_ind, p_value_ind, dof_ind, expected_ind = chi_square_test_of_independence_from_csv(filepath, 'DBend', 'DBend')
        print("Chi-Square Test of Independence:")
        print(f"stat: {stat_ind:.4f}, p-value: {p_value_ind:.4f}, dof: {dof_ind}\n")

        # Fisher's Exact Test example
        odds_ratio, p_value_fisher = fishers_exact_test_from_csv(filepath, 'Binary1', 'Binary2')
        print("Fisher's Exact Test:")
        print(f"odds ratio: {odds_ratio:.4f}, p-value: {p_value_fisher:.4f}\n")

        # Z-test for Proportions example
        z_stat, p_value_z = z_test_for_proportions(30, 100, 0.3)
        print("Z-test for Proportions:")
        print(f"z-statistic: {z_stat:.4f}, p-value: {p_value_z:.4f}\n")

        # Binomial Test example
        p_value_binom = binomial_test_from_csv(filepath, 'BinaryColumn', 0.5)
        print("Binomial Test:")
        print(f"p-value: {p_value_binom:.4f}\n")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
