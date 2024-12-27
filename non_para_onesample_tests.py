import numpy as np
import pandas as pd
from scipy import stats
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import wilcoxon, binom_test

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

def sign_test_from_csv(filepath, column, median):
    """
    Performs a sign test for a single sample using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column (str): The name of the column containing sample data.
    median (float): The hypothesized median of the population.

    Returns:
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

    signs = (sample_data > median).sum()  # Count values above the median
    n = len(sample_data)

    p_value = binom_test(signs, n=n, p=0.5, alternative='two-sided')
    return p_value

def wilcoxon_signed_rank_test_from_csv(filepath, column, median):
    """
    Performs a Wilcoxon signed-rank test for a single sample using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    column (str): The name of the column containing sample data.
    median (float): The hypothesized median of the population.

    Returns:
    stat (float): The test statistic.
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

    stat, p_value = wilcoxon(sample_data - median)
    return stat, p_value

# Example usage
if __name__ == "__main__":
    # Replace 'your_file_path.csv' with the actual path to your CSV file
    filepath = 'sample_dataset.csv'

    try:
        # One-sample t-test example
        t_stat, p_value = one_sample_t_test_from_csv(filepath, 'DGirth', 3.0)
        print("One-sample t-test:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}\n")

        # Sign test example
        p_value_sign = sign_test_from_csv(filepath, 'DGirth', 3.0)
        print("Sign test:")
        print(f"p-value: {p_value_sign:.4f}\n")

        # Wilcoxon signed-rank test example
        stat, p_value_wilcoxon = wilcoxon_signed_rank_test_from_csv(filepath, 'DGirth', 3.0)
        print("Wilcoxon signed-rank test:")
        print(f"stat: {stat:.4f}, p-value: {p_value_wilcoxon:.4f}\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
