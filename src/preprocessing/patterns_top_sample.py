import pandas as pd
import os

# Define base and data directories using relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')

# Define file paths
input_file = os.path.join(data_dir, 'matched_patterns_cleaned.csv')
output_file = os.path.join(data_dir, 'top_patterns_sample.csv')

# Load the matched patterns dataset
matched_patterns_df = pd.read_csv(input_file)

# Create an empty DataFrame to store the top patterns for each solo
top_patterns_df = pd.DataFrame()

# Group by performer and title to extract the top 50 patterns by frequency for each combination
grouped = matched_patterns_df.groupby(['performer', 'title'])

for (performer, title), group in grouped:
    # Sort the group by frequency in descending order and take the top 50
    top_patterns = group.sort_values(by='pattern_frequency', ascending=False).head(50)
    # Append these top patterns to the resulting DataFrame
    top_patterns_df = pd.concat([top_patterns_df, top_patterns], ignore_index=True)

# Sort the resulting DataFrame by pattern_id and reset the index
top_patterns_df = top_patterns_df.sort_values(by='pattern_id').reset_index(drop=True)

# Save the resulting sample to a new CSV file
top_patterns_df.to_csv(output_file, index=False)

print(f"Representative sample saved to {output_file}")