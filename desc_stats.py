import pandas as pd
import matplotlib.pyplot as plt

def compute_descriptive_statistics(file_path):
    data = pd.read_csv(file_path)
    stats = {
        'mean': data.mean(),
        'median': data.median(),
        'mode': data.mode().iloc[0],
        'variance': data.var(),
        'std_dev': data.std(),
        'range': data.max() - data.min(),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'skewness': data.skew(),
        'kurtosis': data.kurt(),
    }
    return pd.DataFrame(stats)





def compute_statistic(column_name, statistic, file_path):
    data = pd.read_csv(file_path)
    if column_name not in data.columns:
        return f"Column '{column_name}' does not exist in the dataset."

    # Select the column
    column = data[column_name]
    if statistic == 'mode':  # Mode
        return column.mode().iloc[0] if not column.mode().empty else None
    # Check if the column is numeric
    if not pd.api.types.is_numeric_dtype(column):
        return f"Column '{column_name}' is not numeric and cannot be analyzed."

    # Compute the requested statistic
    if statistic == 'mean':
        return column.mean()
    elif statistic == 'median':
        return column.median()
    elif statistic == 'std':  # Standard deviation
        return column.std()
    elif statistic == 'var':  # Variance
        return column.var()
    elif statistic == 'range':
        return column.max() - column.min()
    elif statistic == 'iqr':  # Interquartile Range
        return column.quantile(0.75) - column.quantile(0.25)
    elif statistic == 'skewness':
        return column.skew()
    elif statistic == 'kurtosis':
        return column.kurt()
    elif statistic == 'sum':
        return column.sum()
    elif statistic == 'count':
        return column.count()
    elif statistic == 'min':  # Minimum
        return column.min()
    elif statistic == 'max':  # Maximum
        return column.max()
    elif statistic == 'q1':  # 25th percentile
        return column.quantile(0.25)
    elif statistic == 'q3':  # 75th percentile
        return column.quantile(0.75)
    else:
        return f"Statistic '{statistic}' is not supported."

#mode, ft, histogram, min, max, q1, q3
    
def frequency_table(column_name, file_path):
    data = pd.read_csv(file_path)
    if column_name not in data.columns:
        return f"Column '{column_name}' does not exist in the dataset."

    column = data[column_name]

    # Check if the column is categorical
    if pd.api.types.is_numeric_dtype(column):
        return f"Column '{column_name}' is numeric. Frequency tables are typically for categorical data."

    # Generate frequency table
    freq_table = column.value_counts().reset_index()
    freq_table.columns = ['Value', 'Frequency']
    return freq_table
def cross_tabulation(column1, column2, file_path):
    data = pd.read_csv(file_path)
    if column1 not in data.columns or column2 not in data.columns:
        return f"One or both columns '{column1}' and '{column2}' do not exist in the dataset."

    # Generate cross-tabulation
    cross_tab = pd.crosstab(data[column1], data[column2])
    return cross_tab


def plot_histogram(column_name, file_path, bins):
    data = pd.read_csv(file_path)
    if column_name not in data.columns:
        return f"Column '{column_name}' does not exist in the dataset."

    column = data[column_name]

    # Check if the column is numeric
    if not pd.api.types.is_numeric_dtype(column):
        return f"Column '{column_name}' is not numeric. Histograms are for numeric data."

    # Plot histogram
    plt.hist(column.dropna(), bins=bins, edgecolor='black')
    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()





print(compute_statistic("DLength", "mean", "sample_dataset.csv"))
# should be 6.1

print(compute_statistic("DGirth", "median", "sample_dataset.csv"))
# should be 4.5

print(compute_statistic("DBend", "mode", "sample_dataset.csv"))
# should be 'left'

print(compute_statistic("SexQ", "var", "sample_dataset.csv"))
# should be 7.0

print(compute_statistic("DLength", "std", "sample_dataset.csv"))
# should be approximately 1.2

print(compute_statistic("DLength", "range", "sample_dataset.csv"))
# should be 7.9 - 4.0 = 3.9

print(compute_statistic("DLength", "iqr", "sample_dataset.csv"))
# should be Q3 - Q1 = 6.6 - 5.8 = 0.8

print(compute_statistic("SexQ", "skewness", "sample_dataset.csv"))
# should be approximately 0.3

print(compute_statistic("SexQ", "kurtosis", "sample_dataset.csv"))
# should be approximately -1.2

print(compute_statistic("DGirth", "min", "sample_dataset.csv"))
# should be 3.6

print(compute_statistic("DGirth", "max", "sample_dataset.csv"))
# should be 5.4

print(compute_statistic("DLength", "q1", "sample_dataset.csv"))
# should be 5.8

print(compute_statistic("DLength", "q3", "sample_dataset.csv"))
# should be 6.6

print(frequency_table("DBend", "sample_dataset.csv"))
# should return a frequency table with 'left' occurring 7 times, 'down' 3 times, 'straight' 6 times, 'right' 4 times, and 'up' 2 times.

print(cross_tabulation("DBend", "SexQ", "sample_dataset.csv"))
# should return a crosstab with frequencies of combinations of 'DBend' and 'SexQ'.

plot_histogram("DLength", "sample_dataset.csv", bins=5)
# should display a histogram of DLength with 5 bins.
