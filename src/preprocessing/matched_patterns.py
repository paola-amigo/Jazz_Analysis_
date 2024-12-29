import pandas as pd
import ast
import os

# Define base and data directories using relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')

# Define file paths
patterns_file = os.path.join(data_dir, 'Patterns', 'patterns_processed.csv')
solos_file = os.path.join(data_dir, 'solos_db_with_durations_cleaned.csv')
output_file = os.path.join(data_dir, 'matched_patterns.csv')

# Load the patterns and solos_db dataframes
patterns_df = pd.read_csv(patterns_file)
solos_db = pd.read_csv(solos_file)

# Normalize the performer and title fields in both dataframes
patterns_df = patterns_df.copy()
patterns_df.loc[:, 'performer'] = patterns_df['performer'].str.lower().str.replace('[^a-z0-9]', '', regex=True).str.strip()
patterns_df.loc[:, 'title'] = patterns_df['title'].str.lower().str.replace('[^a-z]', '', regex=True).str.strip()  # Remove any numbers

solos_db = solos_db.copy()
solos_db.loc[:, 'performer'] = solos_db['performer'].str.lower().str.replace('[^a-z0-9]', '', regex=True).str.strip()
solos_db.loc[:, 'title'] = solos_db['title'].str.lower().str.replace('[^a-z]', '', regex=True).str.strip()  # Remove any numbers

# Initialize an empty list to store matches
matched_data = []

# Iterate over each row in patterns_df
for index, row in patterns_df.iterrows():
    performer = row['performer']
    title = row['title']
    metrical_position = row['metricalposition']
    values = ast.literal_eval(row['value']) if isinstance(row['value'], str) else row['value']
    num_notes = row['N']

    # Find matches in solos_db
    sample_solo = solos_db[(solos_db['performer'] == performer) & (solos_db['title'] == title)]

    if not sample_solo.empty:
        # Start with the first note by finding the matching metrical position row in the solo data
        matched_metrical_row = sample_solo[sample_solo['metrical_position'] == metrical_position]

        if not matched_metrical_row.empty:
            start_index = matched_metrical_row.index[0]  # Get the index of the first note

            # Iterate over the next 'N' notes to get their details
            for i in range(num_notes):
                current_index = start_index + i

                if current_index in sample_solo.index:
                    match = sample_solo.loc[current_index]

                    matched_data.append({
                        'pattern_id': index,
                        'performer': performer,
                        'title': title,
                        'metrical_position': match['metrical_position'],
                        'sixteenth_representation': match['sixteenth_representation'],
                        'melid': match['melid'],
                        'pitch': match['pitch'],
                        'key': match['key'],
                        # Also include from patterns_df: 'N', 'value'
                        'N': row['N'],
                        'value': values  # Keep the original pattern values
                    })

# Convert matched data to DataFrame
matched_full_df = pd.DataFrame(matched_data)

# Save the matched data to a new CSV file
matched_full_df.to_csv(output_file, index=False)

print(f"Matching complete. Results saved to {output_file}")