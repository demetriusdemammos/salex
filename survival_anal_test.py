import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

def log_rank_test_from_csv(filepath, duration_column, event_column, group_column):
    """
    Performs the Log-Rank Test to compare survival distributions between groups.

    Parameters:
    filepath (str): Path to the CSV file.
    duration_column (str): The column indicating survival times.
    event_column (str): The column indicating event occurrence (1 for event, 0 for censoring).
    group_column (str): The column indicating group membership.

    Returns:
    result (lifelines.StatisticalResult): The result of the Log-Rank Test.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    if duration_column not in data.columns or event_column not in data.columns or group_column not in data.columns:
        raise ValueError(f"Required columns '{duration_column}', '{event_column}', or '{group_column}' not found in the dataset.")

    groups = data[group_column].unique()
    if len(groups) != 2:
        raise ValueError("Log-Rank Test is designed for comparing exactly two groups.")

    group1 = data[data[group_column] == groups[0]]
    group2 = data[data[group_column] == groups[1]]

    result = logrank_test(
        durations_A=group1[duration_column],
        durations_B=group2[duration_column],
        event_observed_A=group1[event_column],
        event_observed_B=group2[event_column]
    )
    return result

def cox_proportional_hazards_model_from_csv(filepath, duration_column, event_column, covariate_columns):
    """
    Fits a Cox Proportional Hazards model to test the effect of covariates on survival times.

    Parameters:
    filepath (str): Path to the CSV file.
    duration_column (str): The column indicating survival times.
    event_column (str): The column indicating event occurrence (1 for event, 0 for censoring).
    covariate_columns (list of str): The columns containing covariates to include in the model.

    Returns:
    cph_summary (DataFrame): Summary of the Cox Proportional Hazards model.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the file path.")

    required_columns = [duration_column, event_column] + covariate_columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.dropna(subset=required_columns)

    if data.empty:
        raise ValueError("No complete cases available for analysis after removing NaN values.")

    cph = CoxPHFitter()
    cph.fit(data[[duration_column, event_column] + covariate_columns], duration_col=duration_column, event_col=event_column)
    return cph.summary

# Example usage
if __name__ == "__main__":
    # Replace 'your_file_path.csv' with the actual path to your CSV file
    filepath = 'sample_dataset.csv'

    try:
        # Log-Rank Test example
        result_logrank = log_rank_test_from_csv(filepath, 'SurvivalTime', 'Event', 'Group')
        print("Log-Rank Test:")
        print(f"Test Statistic: {result_logrank.test_statistic:.4f}, p-value: {result_logrank.p_value:.4f}\n")

        # Cox Proportional Hazards Model example
        cph_summary = cox_proportional_hazards_model_from_csv(filepath, 'SurvivalTime', 'Event', ['Age', 'Treatment'])
        print("Cox Proportional Hazards Model:")
        print(cph_summary)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
