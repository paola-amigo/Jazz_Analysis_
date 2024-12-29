import pandas as pd
import os

# Define base and data directories using relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')

# Define file paths
input_file = os.path.join(data_dir, 'matched_patterns.csv')
output_file = os.path.join(data_dir, 'matched_patterns_cleaned.csv')

# Load the matched patterns dataset
matched_patterns_df = pd.read_csv(input_file)

# Drop rows with missing 'key'
matched_patterns_df = matched_patterns_df.dropna(subset=['key'])

# Sort by pattern_id to ensure we keep the first occurrence in the case of duplicates
matched_patterns_df = matched_patterns_df.sort_values(by=['pattern_id'])

# Identify the first pattern_id for each unique combination of performer, title, and value
first_occurrences = matched_patterns_df.drop_duplicates(subset=['performer', 'title', 'value'], keep='first')

# Extract the pattern_ids that we want to keep
valid_pattern_ids = first_occurrences['pattern_id'].unique()

# Keep only rows with the selected pattern_ids, ensuring all rows of the first occurrence are kept
matched_patterns_df = matched_patterns_df[matched_patterns_df['pattern_id'].isin(valid_pattern_ids)]

# Calculate the frequency of each unique pattern value across different pattern_ids
# Group by 'value' and count the number of unique 'pattern_id' for each 'value'
pattern_frequency = matched_patterns_df.groupby('value')['pattern_id'].nunique()

# Map the calculated pattern frequency back to the DataFrame
matched_patterns_df['pattern_frequency'] = matched_patterns_df['value'].map(pattern_frequency)

# Save the cleaned dataset to a new CSV file
matched_patterns_df.to_csv(output_file, index=False)

print(f"Cleaned patterns saved to {output_file}")