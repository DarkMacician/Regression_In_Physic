import pandas as pd

# Load the first dataset
file1 = "D:/Python Project/RecommendationSystem/bsc_gas_transaction_data.csv"
df1 = pd.read_csv(file1)

# Load the second dataset
file2 = "D:/Python Project/RecommendationSystem/tradingbot/updated_data.csv"
df2 = pd.read_csv(file2)

# Ensure timestamps are in datetime format for both datasets
df1['Timestamp'] = pd.to_datetime(df1['Timestamp'])
df2['Timestamp'] = pd.to_datetime(df2['Timestamp'])

# Merge the two datasets on the Timestamp column
# Use 'inner' join to include only matching timestamps in both datasets
merged_df = pd.merge(df1, df2, on='Timestamp', how='inner')

# Save the merged dataset to a new CSV file
output_file = "merged_dataset.csv"
merged_df.to_csv(output_file, index=False)

print(f"Merged dataset saved to {output_file}")
