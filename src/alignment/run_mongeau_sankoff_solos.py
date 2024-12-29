import sys
import os
import time
import numpy as np
import pandas as pd
import itertools
# Add directory to find function
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.alignment.mongeau_sankoff_functions import mongeau_sankoff_alignment

input_file_path = os.path.join(os.path.dirname(__file__), '../../data/sequences_solos.csv')


# Load the full solo dataset
sequences_df = pd.read_csv(input_file_path)
print(f"Loaded {len(sequences_df)} rows from the dataset.")

# Define alignment parameter of mongeau_sankoff_function
k2= 0.75

# Validate sequences format
sequences = sequences_df['sequence_tuples'].apply(eval)

# Sample of tuples to validate results
max_tuples = 50
subsampled_sequences = sequences.apply(lambda seq: seq[:max_tuples])

# Create combinations of the subsampled sequences along with their IDs for pairwise alignment
sequence_pairs = list(itertools.combinations(zip(subsampled_sequences, sequences_df['melid']), 2))

# Initiate empty to store results
aligned_results = []

# Store  maximum alignment score found
max_alignment_score = 0

# Total time for all pairs
total_run_start_time = time.time()

# Iteration through all sequence pairs for alignment to find the maximum alignment score
for index, ((seq1, seq1_id), (seq2, seq2_id)) in enumerate(sequence_pairs):  
    total_start_time = time.time()
    print(f"\nProcessing alignment for pair {index + 1} of {len(sequence_pairs)}")
    
    # Run the Mongeau-Sankoff alignment
    alignment_quality = mongeau_sankoff_alignment(seq1, seq2,k2)

    # Total time for processing the pair
    total_elapsed_time = time.time() - total_start_time

    # Print summarized timing information for each pair
    print(f"Alignment for pair {index + 1} completed in {total_elapsed_time:.2f} seconds")

    # Update the max alignment score found
    if alignment_quality > max_alignment_score:
        max_alignment_score = alignment_quality

    # Add the result along with sequence IDs for tracking
    aligned_results.append({
        'sequence_1_id': seq1_id,
        'sequence_2_id': seq2_id,
        'alignment_quality': alignment_quality
    })

# Calculate the total elapsed time for all pairs
total_run_elapsed_time = time.time() - total_run_start_time

# Print the total elapsed time
print(f"\nTotal time for processing all pairs: {total_run_elapsed_time:.2f} seconds")

# Convert and save normalised results into a DataFrame
aligned_results_df = pd.DataFrame(aligned_results)
aligned_results_df['similarity'] = aligned_results_df['alignment_quality'] / max_alignment_score
output_file_path = os.path.join(os.path.dirname(__file__), f'../../data/output/aligned_solos_results_k2_{k2:.2f}.csv')
aligned_results_df.to_csv(output_file_path, index=False)
print(f"Alignment results saved to {output_file_path}")

# Create similarity matrix 
unique_ids = sequences_df['melid'].unique()
similarity_matrix = pd.DataFrame(np.zeros((len(unique_ids), len(unique_ids))), index=unique_ids, columns=unique_ids)

# Fill similarity matrix
for _, row in aligned_results_df.iterrows():
    similarity_matrix.loc[row['sequence_1_id'], row['sequence_2_id']] = row['similarity']
    similarity_matrix.loc[row['sequence_2_id'], row['sequence_1_id']] = row['similarity']

# Fill diagonal with similarity = 1 for self-alignment
np.fill_diagonal(similarity_matrix.values, 1)

# Save the similarity matrix as CSV
similarity_matrix_file_path = os.path.join(os.path.dirname(__file__), f'../../data/output/similarity_matrix_solos_k2_{k2:.2f}.csv')
similarity_matrix.to_csv(similarity_matrix_file_path)
print(f"Similarity matrix saved to {similarity_matrix_file_path}")