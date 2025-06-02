import pandas as pd

# Define the file path
file_path = "/home/manuel/Documents/AI/pogoh-ai-engineering/data/raw/april-2025.xlsx"

# Load the Excel file into a DataFrame
pogoh_df = pd.read_excel(file_path)

# Display basic info and the first few rows
print(pogoh_df.info())
print(pogoh_df.head())

# Summary statistics for numeric columns
summary_stats = pogoh_df.describe()

# Count of unique values per column
unique_counts = pogoh_df.nunique()

# Count of missing values per column
missing_values = pogoh_df.isnull().sum()

# Value counts for key categorical variables
closed_status_counts = pogoh_df["Closed Status"].value_counts()
rider_type_counts = pogoh_df["Rider Type"].value_counts()

# Print outputs
print("=== Summary Statistics ===")
print(summary_stats)

print("\n=== Unique Value Counts ===")
print(unique_counts)

print("\n=== Missing Values ===")
print(missing_values)

print("\n=== Closed Status Distribution ===")
print(closed_status_counts)

print("\n=== Rider Type Distribution ===")
print(rider_type_counts)

pogoh_df["Closed Status"].unique()
pogoh_df["Rider Type"].unique()