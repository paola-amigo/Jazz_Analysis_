import pandas as pd
import itertools
import time
import os
import sys
import numpy as np

# Add directory to find function
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.alignment.mongeau_sankoff_functions import mongeau_sankoff_alignment

# Define alignment parameter of mongeau_sankoff_function
k2 = 0.75 

# Input and output paths
sequences_file_path = os.path.join(os.path.dirname(__file__), '../../data/sequences_patterns.csv')
output_dir = os.path.join(os.path.dirname(__file__), '../../data/output/')
os.makedirs(output_dir, exist_ok=True)

# Load the sequences dataset
sequences_df = pd.read_csv(sequences_file_path)

# Step 1: Validate structure of input data
if 'pattern_id' not in sequences_df.columns or 'sequence_tuples' not in sequences_df.columns:
    raise ValueError("Input file must contain 'pattern_id' and 'sequence_tuples' columns.")

try:
    sequences_df['sequence_tuples'] = sequences_df['sequence_tuples'].apply(eval)
except Exception as e:
    raise ValueError(f"Error parsing sequence tuples: {e}")

# Check for duplicate pattern IDs
if sequences_df['pattern_id'].duplicated().any():
    raise ValueError("Duplicate pattern IDs detected in input data.")

print(f"Loaded {len(sequences_df)} patterns from sequences_patterns.csv")
print(f"Unique pattern IDs: {sequences_df['pattern_id'].nunique()}")

# Drop rows with null sequences
if sequences_df['sequence_tuples'].isnull().any():
    print("Warning: Null sequences detected. These will be removed.")
    sequences_df = sequences_df.dropna(subset=['sequence_tuples'])

# Step 2: Create combinations of the sequences for pairwise alignment
sequence_pairs = list(itertools.combinations(zip(sequences_df['sequence_tuples'], sequences_df['pattern_id']), 2))
print(f"Generated {len(sequence_pairs)} sequence pairs for alignment.")

aligned_results = []  # To store results
max_alignment_score = 0  # Track the max alignment score

total_run_start_time = time.time()

# Step 3: Process each pair
for index, ((seq1, seq1_id), (seq2, seq2_id)) in enumerate(sequence_pairs):
    print(f"Processing alignment for pair {index + 1} of {len(sequence_pairs)}")

    start_time = time.time()
    try:
        alignment_quality = mongeau_sankoff_alignment(seq1, seq2, k2)
    except Exception as e:
        print(f"Error processing pair {seq1_id}-{seq2_id}: {e}")
        continue
    
    elapsed_time = time.time() - start_time
    print(f"Processed pair {index + 1} in {elapsed_time:.2f} seconds")
    
    max_alignment_score = max(max_alignment_score, alignment_quality)

    aligned_results.append({
        'pattern_id_1': seq1_id,
        'pattern_id_2': seq2_id,
        'alignment_quality': alignment_quality
    })

# Validate alignment results
if not aligned_results:
    raise ValueError("No alignment results generated. Check input data and alignment function.")

aligned_results_df = pd.DataFrame(aligned_results)

# Step 4: Normalise and save alignment results
aligned_results_df['similarity'] = aligned_results_df['alignment_quality'] / max_alignment_score
output_file_path = os.path.join(output_dir, f'aligned_patterns_results_k2_{k2:.2f}.csv')
aligned_results_df.to_csv(output_file_path, index=False)
print(f"Alignment results saved to {output_file_path}")

# Validate coverage of pattern IDs in results
result_pattern_ids = set(aligned_results_df['pattern_id_1']).union(set(aligned_results_df['pattern_id_2']))
missing_pattern_ids = set(sequences_df['pattern_id']) - result_pattern_ids
if missing_pattern_ids:
    print(f"Warning: Missing pattern IDs in alignment results: {missing_pattern_ids}")

# Step 5: Create and validate the similarity matrix
unique_ids = sequences_df['pattern_id'].unique()
similarity_matrix = pd.DataFrame(np.zeros((len(unique_ids), len(unique_ids))), index=unique_ids, columns=unique_ids)

# Populate similarity matrix
for _, row in aligned_results_df.iterrows():
    similarity_matrix.loc[row['pattern_id_1'], row['pattern_id_2']] = row['similarity']
    similarity_matrix.loc[row['pattern_id_2'], row['pattern_id_1']] = row['similarity']

np.fill_diagonal(similarity_matrix.values, 1)  # Self-alignment is 1

# Check for null values in the matrix
if similarity_matrix.isnull().values.any():
    print("Warning: Similarity matrix contains null values.")
    print(similarity_matrix)

similarity_matrix_file_path = os.path.join(output_dir, f'similarity_matrix_patterns_k2_{k2:.2f}.csv')
similarity_matrix.to_csv(similarity_matrix_file_path)
print(f"Similarity matrix saved to {similarity_matrix_file_path}")

# Step 6: Validate similarity matrix dimensions
if similarity_matrix.shape[0] != len(unique_ids) or similarity_matrix.shape[1] != len(unique_ids):
    raise ValueError("Mismatch in similarity matrix dimensions.")

print("Alignment process completed successfully.")