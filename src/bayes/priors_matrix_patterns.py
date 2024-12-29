import numpy as np
import pandas as pd

# Load similarity matrix to get aligned pattern IDs
similarity_matrix_df = pd.read_csv('data/output/similarity_matrix_patterns_k2_0.75.csv', index_col=0)
aligned_pattern_ids = similarity_matrix_df.index.astype(int).unique()  # Use the unnamed column as index
print(f"Aligned Pattern IDs: {aligned_pattern_ids}")

# Load top_patterns_sample to get valid pattern IDs and corresponding melids
top_patterns_sample_df = pd.read_csv('data/top_patterns_sample.csv')

# Filter top_patterns_sample to only include the aligned pattern IDs
filtered_top_patterns = top_patterns_sample_df[top_patterns_sample_df['pattern_id'].isin(aligned_pattern_ids)]
print(f"Filtered Top Patterns Shape: {filtered_top_patterns.shape}")

# Extract the required columns: pattern_id and melid
aligned_patterns_with_melids = filtered_top_patterns[['pattern_id', 'melid']].drop_duplicates()

# Load solos dataset to get metadata for the aligned melid values
solos_df = pd.read_csv('data/solos_db_with_durations_cleaned.csv')

# Filter solos dataset to only include the aligned melid values
filtered_solos = solos_df[solos_df['melid'].isin(aligned_patterns_with_melids['melid'])]
print(f"Filtered Solos Shape: {filtered_solos.shape}")

# Extract relevant columns: melid, performer, title, release_date
filtered_solos = filtered_solos[['melid', 'performer', 'title', 'releasedate']]

# Sort the solos based on release date to maintain chronological order
filtered_solos.sort_values(by='releasedate', inplace=True)

# Create a dictionary to map melid to release date for easy lookup
melid_to_release_date = filtered_solos.set_index('melid')['releasedate'].to_dict()

# Create a mapping from pattern_id to melid using aligned_patterns_with_melids
pattern_id_to_melid = aligned_patterns_with_melids.set_index('pattern_id')['melid'].to_dict()

# Create an empty prior matrix based on the number of aligned patterns
valid_pattern_ids = aligned_patterns_with_melids['pattern_id'].unique()
num_aligned_patterns = len(valid_pattern_ids)
prior_matrix = np.zeros((num_aligned_patterns, num_aligned_patterns))

# Create a mapping from pattern_id to index in the prior matrix
pattern_id_to_index = {pattern_id: i for i, pattern_id in enumerate(valid_pattern_ids)}

# Iterate through each pair of pattern_ids to determine temporal influence based on melid release dates
for pattern_id_a in valid_pattern_ids:
    melid_a = pattern_id_to_melid[pattern_id_a]
    release_date_a = melid_to_release_date[melid_a]

    for pattern_id_b in valid_pattern_ids:
        melid_b = pattern_id_to_melid[pattern_id_b]
        release_date_b = melid_to_release_date[melid_b]

        # If melid_a has an earlier release date than melid_b, it can be an influence
        if release_date_a < release_date_b:
            prior_matrix[pattern_id_to_index[pattern_id_a], pattern_id_to_index[pattern_id_b]] = 1

print(f"Prior Matrix Shape: {prior_matrix.shape}")

# Save the prior matrix to a file
prior_matrix_path = 'data/output/prior_matrix_patterns.npy'
np.save(prior_matrix_path, prior_matrix)

print(f"Prior matrix saved to {prior_matrix_path}")
