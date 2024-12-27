import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def one_way_anova(*groups):
    """
    Performs a one-way ANOVA test.

    Parameters:
    groups (arrays): Arrays of data for each group.

    Returns:
    f_stat (float): The F-statistic.
    p_value (float): The p-value of the test.
    """
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value

def two_way_anova_from_csv(filepath, dependent_var, factor1, factor2):
    """
    Performs a two-way ANOVA test using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    dependent_var (str): The name of the dependent variable.
    factor1 (str): The name of the first independent variable (factor).
    factor2 (str): The name of the second independent variable (factor).

    Returns:
    anova_table (DataFrame): The ANOVA table with F-statistics and p-values.
    """
    # Verify file exists
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    # User controls: Ensure necessary columns exist
    required_columns = [dependent_var, factor1, factor2]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Clean data to remove NaN and infinite values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure factors are categorical
    data[factor1] = data[factor1].astype('category')
    data[factor2] = data[factor2].astype('category')

    # Check for empty groups or unbalanced data
    if data.empty:
        raise ValueError("Dataset is empty after cleaning. Please ensure data integrity.")

    model = ols(f'{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def repeated_measures_anova_from_csv(filepath, subject, dependent_var, within_factor):
    """
    Performs a repeated measures ANOVA using data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    subject (str): The name of the subject identifier.
    dependent_var (str): The name of the dependent variable.
    within_factor (str): The name of the within-subject factor.

    Returns:
    anova_table (DataFrame): The ANOVA table with F-statistics and p-values.
    """
    # Verify file exists
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    # User controls: Ensure necessary columns exist
    required_columns = [subject, dependent_var, within_factor]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Clean data to remove NaN and infinite values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure factors are categorical
    data[within_factor] = data[within_factor].astype('category')

    # Check for empty groups or unbalanced data
    if data.empty:
        raise ValueError("Dataset is empty after cleaning. Please ensure data integrity.")

    model = ols(f'{dependent_var} ~ C({within_factor}) + C({subject})', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

# Example usage
if __name__ == "__main__":
    # One-way ANOVA example
    group1 = [5.1, 5.3, 5.5, 5.7, 5.9]
    group2 = [6.1, 6.3, 6.5, 6.7, 6.9]
    group3 = [7.1, 7.3, 7.5, 7.7, 7.9]

    f_stat, p_value = one_way_anova(group1, group2, group3)
    print("One-way ANOVA:")
    print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}\n")

    # Two-way ANOVA example
    try:
        # Replace 'your_file_path.csv' with the actual path to your CSV file
        anova_table = two_way_anova_from_csv('sample_dataset.csv', 'SexQ', 'DGirth', 'DBend')
        print("Two-way ANOVA:")
        print(anova_table)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

    # Repeated measures ANOVA example
    try:
        # Replace 'your_file_path.csv' with the actual path to your CSV file
        repeated_anova_table = repeated_measures_anova_from_csv('sample_dataset.csv', 'num', 'SexQ', 'DGirth')
        print("\nRepeated Measures ANOVA:")
        print(repeated_anova_table)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

