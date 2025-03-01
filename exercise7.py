import pandas as pd
import numpy as np

# Load the Excel file
file_path = "Question1_Final_CP[2].xlsx"
xls = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
xls.sheet_names
# Load the sheet into a DataFrame
df = pd.read_excel(xls, sheet_name="Лист1")

# Display the first few rows to understand the data structure
df.head()
# Extract the relevant column for analysis
values = df["Variable"]

# Compute mean for SRS
mean_srs = np.mean(values)

# Compute standard error for SRS
std_srs = np.std(values, ddof=1)  # Sample standard deviation
n_srs = len(values)  # Sample size
se_srs = std_srs / np.sqrt(n_srs)  # Standard Error

# Compute 95% confidence interval
t_value = 2.04  # Given in the problem
ci_upper_srs = mean_srs + t_value * se_srs
ci_lower_srs = mean_srs - t_value * se_srs

# Round results to 2 decimal places
mean_srs, se_srs, ci_upper_srs, ci_lower_srs = round(mean_srs, 2), round(se_srs, 2), round(ci_upper_srs, 3), round(ci_lower_srs, 3)

mean_srs, se_srs, ci_upper_srs, ci_lower_srs



# # Compute mean for Clustered Random Sampling
# mean_clustered = np.mean(values)  # Mean remains the same as in SRS

# # Compute standard error for Clustered Sampling by treating clusters separately
# clusters = df["Cluster"].unique()
# cluster_means = df.groupby("Cluster")["Variable"].mean()
# num_clusters = len(clusters)
# std_clustered = np.std(cluster_means, ddof=1)  # Sample standard deviation of cluster means
# se_clustered = std_clustered / np.sqrt(num_clusters)  # Standard Error for clustering

# # Compute d-value (ratio of standard errors)
# d_value = se_clustered / se_srs

# # Compute d-squared
# d_squared = d_value ** 2

# # Compute roh (intra-cluster correlation coefficient)
# Wcl = 0.125  # Given in the problem
# roh = (d_squared - Wcl) / (1 - Wcl)

# # Compute Neff (effective sample size)
# Neff = n_srs / (1 + (n_srs - 1) * roh)

# # Round results to 2 decimal places
# mean_clustered, se_clustered, d_value, d_squared, roh, Neff = (
#     round(mean_clustered, 2),
#     round(se_clustered, 2),
#     round(d_value, 2),
#     round(d_squared, 2),
#     round(roh, 2),
#     round(Neff, 2),
# )

# mean_clustered, se_clustered, d_value, d_squared, roh, Neff
