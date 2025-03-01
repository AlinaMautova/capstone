import pandas as pd

# Load the dataset
file_path = "Question7_Final_CP[1].csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
df.head()

# Define the required age groups
target_age_groups = ["18-24", "55-64"]

# Filter the dataset for the required age groups
df_filtered = df[df["Age Group"].isin(target_age_groups)]

# Count the occurrences of each age group
age_counts = df_filtered["Age Group"].value_counts()

# Total sample size (assuming the dataset represents the entire sample)
total_sample_size = len(df)

# Compute proportions
age_proportions = age_counts / total_sample_size

# Define the category-specific parameters
category_specific_params_age = {
    "18-24": {"rho": 0.020, "m": 5},
    "55-64": {"rho": 0.018, "m": 5},
}

# Compute standard errors, confidence intervals, and adjusted values
results = []

for age_group, count in age_counts.items():
    p = age_proportions[age_group]  # Proportion
    n = count  # Sample size for the age group
    SE = (p * (1 - p) / n) ** 0.5  # Standard Error

    # 95% CI
    Z = 1.96  # For 95% confidence level
    CI_lower = p - Z * SE
    CI_upper = p + Z * SE

    # Compute Design Effect (DEFF)
    rho = category_specific_params_age[age_group]["rho"]
    m = category_specific_params_age[age_group]["m"]
    DEFF = 1 + rho * (m - 1)

    # Adjusted SE
    SE_adj = SE * (DEFF ** 0.5)

    # Adjusted 95% CI
    CI_adj_lower = p - Z * SE_adj
    CI_adj_upper = p + Z * SE_adj

    # Store results
    results.append([age_group, n, p, SE, SE_adj, (CI_lower, CI_upper), (CI_adj_lower, CI_adj_upper), DEFF])

# Convert results to a DataFrame
columns = ["Age Group", "n", "Estimated Proportion", "Standard Error", "Adjusted SE", "95% CI", "95% CI (Adjusted)", "Design Effect"]
results_df = pd.DataFrame(results, columns=columns)

# Display the computed results
results_df
