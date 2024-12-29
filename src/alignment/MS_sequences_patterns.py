import pandas as pd
import os

# Define directory + Input/Output paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data')
output_path = os.path.join(data_dir, 'sequences_patterns.csv')
top_patterns_file_path = os.path.join(os.path.dirname(__file__), '../../data/top_patterns_sample.csv')

# Load data
matched_patterns = pd.read_csv(os.path.join(data_dir, 'matched_patterns_cleaned.csv'))
top_patterns_sample = pd.read_csv(top_patterns_file_path)

# Filter by columns needed
base_sequences = matched_patterns[['pattern_id', 'melid', 'pitch', 'sixteenth_representation', 'key', 'value']]
print(top_patterns_sample['pattern_id'].dtype)
print(base_sequences['pattern_id'].dtype)

# Filter base_sequences using top_patterns_sample
top_patterns_sample['pattern_id'] = top_patterns_sample['pattern_id'].astype(str)
base_sequences['pattern_id'] = base_sequences['pattern_id'].astype(str)

filtered_base_sequences = base_sequences[base_sequences['pattern_id'].isin(top_patterns_sample['pattern_id'])]
base_sequences = filtered_base_sequences

# Define a dictionary for tonic pitch classes (including modes and chromatic)
TONICS = {
    'C-maj': 60, 'C-min': 60, 'C-dor': 60, 'C-phry': 60, 'C-lyd': 60, 'C-mix': 60, 'C-aeo': 60, 'C-loc': 60, 'C-chrom': 60,
    'C#-maj': 61, 'C#-min': 61, 'C#-dor': 61, 'C#-phry': 61, 'C#-lyd': 61, 'C#-mix': 61, 'C#-aeo': 61, 'C#-loc': 61, 'C#-chrom': 61,
    'Db-maj': 61, 'Db-min': 61, 'Db-dor': 61, 'Db-phry': 61, 'Db-lyd': 61, 'Db-mix': 61, 'Db-aeo': 61, 'Db-loc': 61, 'Db-chrom': 61,
    'D-maj': 62, 'D-min': 62, 'D-dor': 62, 'D-phry': 62, 'D-lyd': 62, 'D-mix': 62, 'D-aeo': 62, 'D-loc': 62, 'D-chrom': 62,
    'Eb-maj': 63, 'Eb-min': 63, 'Eb-dor': 63, 'Eb-phry': 63, 'Eb-lyd': 63, 'Eb-mix': 63, 'Eb-aeo': 63, 'Eb-loc': 63, 'Eb-chrom': 63,
    'E-maj': 64, 'E-min': 64, 'E-dor': 64, 'E-phry': 64, 'E-lyd': 64, 'E-mix': 64, 'E-aeo': 64, 'E-loc': 64, 'E-chrom': 64,
    'F-maj': 65, 'F-min': 65, 'F-dor': 65, 'F-phry': 65, 'F-lyd': 65, 'F-mix': 65, 'F-aeo': 65, 'F-loc': 65, 'F-chrom': 65,
    'F#-maj': 66, 'F#-min': 66, 'F#-dor': 66, 'F#-phry': 66, 'F#-lyd': 66, 'F#-mix': 66, 'F#-aeo': 66, 'F#-loc': 66, 'F#-chrom': 66,
    'Gb-maj': 66, 'Gb-min': 66, 'Gb-dor': 66, 'Gb-phry': 66, 'Gb-lyd': 66, 'Gb-mix': 66, 'Gb-aeo': 66, 'Gb-loc': 66, 'Gb-chrom': 66,
    'G-maj': 67, 'G-min': 67, 'G-dor': 67, 'G-phry': 67, 'G-lyd': 67, 'G-mix': 67, 'G-aeo': 67, 'G-loc': 67, 'G-chrom': 67, 'Ab':68,
    'Ab-maj': 68, 'Ab-min': 68, 'Ab-dor': 68, 'Ab-phry': 68, 'Ab-lyd': 68, 'Ab-mix': 68, 'Ab-aeo': 68, 'Ab-loc': 68, 'Ab-chrom': 68,
    'A-maj': 69, 'A-min': 69, 'A-dor': 69, 'A-phry': 69, 'A-lyd': 69, 'A-mix': 69, 'A-aeo': 69, 'A-loc': 69, 'A-chrom': 69,
    'Bb-maj': 70, 'Bb-min': 70, 'Bb-dor': 70, 'Bb-phry': 70, 'Bb-lyd': 70, 'Bb-mix': 70, 'Bb-aeo': 70, 'Bb-loc': 70, 'Bb-chrom': 70,
    'B-maj': 71, 'B-min': 71, 'B-dor': 71, 'B-phry': 71, 'B-lyd': 71, 'B-mix': 71, 'B-aeo': 71, 'B-loc': 71, 'B-chrom': 71
}

# Function to calculate distance from pitch to the tonic
def calculate_intervals_from_tonic(row):
    tonic = TONICS[row['key']] 
    if isinstance(row['pitch'], list):  
        intervals = [(pitch - tonic) % 12 for pitch in row['pitch']]
    else:  # Handle single pitch
        intervals = (row['pitch'] - tonic) % 12
    
    # Adjust intervals for the shortest path after applying mmodulus 12
    if isinstance(intervals, list):
        intervals = [interval - 12 if interval > 6 else interval for interval in intervals]
    else:
        intervals = intervals - 12 if intervals > 6 else intervals
    
    return intervals

# Validate formats
base_sequences['pitch'] = pd.to_numeric(base_sequences['pitch'], errors='coerce').astype(int)
base_sequences['sixteenth_representation'] = pd.to_numeric(base_sequences['sixteenth_representation'], errors='coerce').astype(int)

# Apply function
base_sequences['interval'] = base_sequences.apply(calculate_intervals_from_tonic, axis=1)

# Group by 'pattern_id' to create sequences as tuples
def create_pattern_sequence(group):
    # Generate a list of (interval, sixteenth_representation) tuples for each pattern
    return [(int(interval), int(sixteenth)) for interval, sixteenth in zip(group['interval'], group['sixteenth_representation'])]

# Create sequences by grouping by 'pattern_id' and applying the function
sequences = (
    base_sequences.groupby('pattern_id')[['interval', 'sixteenth_representation']]
    .apply(create_pattern_sequence)
    .reset_index()
)

sequences.rename(columns={0: 'sequence_tuples'}, inplace=True)

# Merge the generated sequences with unique metadata for future analysis
metadata_columns = ['pattern_id', 'melid', 'key', 'value']
sequences = sequences.merge(base_sequences[metadata_columns].drop_duplicates(), on='pattern_id')

# Save the sequences into a CSV for further use
sequences.to_csv(output_path, index=False)
print(f"Sequences with intervals saved to {output_path}")