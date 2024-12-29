import numpy as np
import pandas as pd
import os

# Define relative paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data')
output_dir = os.path.join(data_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

prior_matrix_output_path = os.path.join(output_dir, 'prior_matrix_solos.npy')

# Load the DataFrame from the CSV file
solos_df = pd.read_csv(os.path.join(data_dir, 'solos_db_with_durations_cleaned.csv'))

# Extract relevant columns: melid, performer, title, release_date
solos_df = solos_df[['melid', 'performer', 'title', 'releasedate']]

# Sort the solos based on release date to maintain chronological order
solos_df.sort_values(by='releasedate', inplace=True)

# Create a dictionary to map melid to release date for easy lookup
melid_to_release_date = solos_df.set_index('melid')['releasedate'].to_dict()

# Create a list of unique melid values
valid_melids = solos_df['melid'].unique()
num_valid_solos = len(valid_melids)

# Create an empty prior matrix based on the number of valid solos
prior_matrix = np.zeros((num_valid_solos, num_valid_solos))

# Create a mapping from melid to index in the prior matrix
melid_to_index = {melid: i for i, melid in enumerate(valid_melids)}

# Iterate through each pair of melid values to determine temporal influence
for melid_a in valid_melids:
    release_date_a = melid_to_release_date[melid_a]

    for melid_b in valid_melids:
        release_date_b = melid_to_release_date[melid_b]

        # If melid_a has an earlier release date than melid_b, it can be an influence
        if release_date_a < release_date_b:
            prior_matrix[melid_to_index[melid_a], melid_to_index[melid_b]] = 1

print(f"Prior Matrix Shape: {prior_matrix.shape}")

# Save the prior matrix to a file
np.save(prior_matrix_output_path, prior_matrix)

print(f"Prior matrix saved to {prior_matrix_output_path}")