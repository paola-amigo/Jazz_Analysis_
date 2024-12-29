import pandas as pd
import ast
import os

# Define directory + Input/Output paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data')
output_path = os.path.join(data_dir, 'sequences_solos.csv')

# Load data
solos_data = pd.read_csv(os.path.join(data_dir, 'solos_db_with_durations_cleaned.csv'))

# Filter by columns needed
full_solos_sequences = solos_data[['melid', 'pitch', 'calculated_duration', 'key']].copy()

# Helper function evaluate pitch
def safe_eval_pitch(pitch):
    try:
        if isinstance(pitch, str):
            return [float(p) for p in ast.literal_eval(pitch)]
        elif isinstance(pitch, (float, int)):
            return [float(pitch)]
        elif isinstance(pitch, list):
            return [float(p) for p in pitch]
        else:
            return []
    except Exception as e:
        print(f"Error parsing pitch: {pitch}. Error: {e}")
        return []

# Apply evaluation function
full_solos_sequences['pitch'] = full_solos_sequences['pitch'].apply(safe_eval_pitch)

# Define a dictionary for tonic pitch classes (including modes and chromatic)
TONICS = {
    'C-maj': 60, 'C-min': 60, 'C-dor': 60, 'C-phry': 60, 'C-lyd': 60, 'C-mix': 60, 'C-aeo': 60, 'C-loc': 60, 'C-chrom': 60, 'C-blues': 60,
    'C#-maj': 61, 'C#-min': 61, 'C#-dor': 61, 'C#-phry': 61, 'C#-lyd': 61, 'C#-mix': 61, 'C#-aeo': 61, 'C#-loc': 61, 'C#-chrom': 61, 'C#-blues': 61,
    'Db-maj': 61, 'Db-min': 61, 'Db-dor': 61, 'Db-phry': 61, 'Db-lyd': 61, 'Db-mix': 61, 'Db-aeo': 61, 'Db-loc': 61, 'Db-chrom': 61, 'Db-blues': 61,
    'D-maj': 62, 'D-min': 62, 'D-dor': 62, 'D-phry': 62, 'D-lyd': 62, 'D-mix': 62, 'D-aeo': 62, 'D-loc': 62, 'D-chrom': 62, 'D-blues': 62,
    'D#-maj': 63, 'D#-min': 63, 'D#-dor': 63, 'D#-phry': 63, 'D#-lyd': 63, 'D#-mix': 63, 'D#-aeo': 63, 'D#-loc': 63, 'D#-chrom': 63, 'D#-blues': 63,
    'Eb-maj': 63, 'Eb-min': 63, 'Eb-dor': 63, 'Eb-phry': 63, 'Eb-lyd': 63, 'Eb-mix': 63, 'Eb-aeo': 63, 'Eb-loc': 63, 'Eb-chrom': 63, 'Eb-blues': 63,
    'E-maj': 64, 'E-min': 64, 'E-dor': 64, 'E-phry': 64, 'E-lyd': 64, 'E-mix': 64, 'E-aeo': 64, 'E-loc': 64, 'E-chrom': 64, 'E-blues': 64,
    'F-maj': 65, 'F-min': 65, 'F-dor': 65, 'F-phry': 65, 'F-lyd': 65, 'F-mix': 65, 'F-aeo': 65, 'F-loc': 65, 'F-chrom': 65, 'F-blues': 65,
    'F#-maj': 66, 'F#-min': 66, 'F#-dor': 66, 'F#-phry': 66, 'F#-lyd': 66, 'F#-mix': 66, 'F#-aeo': 66, 'F#-loc': 66, 'F#-chrom': 66, 'F#-blues': 66,
    'Gb-maj': 66, 'Gb-min': 66, 'Gb-dor': 66, 'Gb-phry': 66, 'Gb-lyd': 66, 'Gb-mix': 66, 'Gb-aeo': 66, 'Gb-loc': 66, 'Gb-chrom': 66, 'Gb-blues': 66,
    'G-maj': 67, 'G-min': 67, 'G-dor': 67, 'G-phry': 67, 'G-lyd': 67, 'G-mix': 67, 'G-aeo': 67, 'G-loc': 67, 'G-chrom': 67, 'G-blues': 67,
    'Ab-maj': 68, 'Ab-min': 68, 'Ab-dor': 68, 'Ab-phry': 68, 'Ab-lyd': 68, 'Ab-mix': 68, 'Ab-aeo': 68, 'Ab-loc': 68, 'Ab-chrom': 68, 'Ab-blues': 68, 'Ab': 68,
    'A-maj': 69, 'A-min': 69, 'A-dor': 69, 'A-phry': 69, 'A-lyd': 69, 'A-mix': 69, 'A-aeo': 69, 'A-loc': 69, 'A-chrom': 69, 'A-blues': 69,
    'Bb-maj': 70, 'Bb-min': 70, 'Bb-dor': 70, 'Bb-phry': 70, 'Bb-lyd': 70, 'Bb-mix': 70, 'Bb-aeo': 70, 'Bb-loc': 70, 'Bb-chrom': 70, 'Bb-blues': 70,
    'B-maj': 71, 'B-min': 71, 'B-dor': 71, 'B-phry': 71, 'B-lyd': 71, 'B-mix': 71, 'B-aeo': 71, 'B-loc': 71, 'B-chrom': 71, 'B-blues': 71,
}

# Function to calculate distance from pitch to the tonic
def calculate_intervals_from_tonic(row):
    tonic = TONICS[row['key']]  # Assume the key is valid and exists in the dictionary
    if isinstance(row['pitch'], list):  # Handle lists of pitches
        intervals = [(pitch - tonic) % 12 for pitch in row['pitch']]
    else:  # Handle single pitch
        intervals = (row['pitch'] - tonic) % 12
    
    # Adjust intervals for the shortest path after applying mmodulus 12
    if isinstance(intervals, list):
        intervals = [interval - 12 if interval > 6 else interval for interval in intervals]
    else:
        intervals = intervals - 12 if intervals > 6 else intervals
    
    return intervals

# Apply function
full_solos_sequences['intervals'] = full_solos_sequences.apply(calculate_intervals_from_tonic, axis=1)

# Convert duration to sixteenths and validate format
full_solos_sequences['sixteenth_duration'] = full_solos_sequences['calculated_duration'].apply(
    lambda x: int(round(16 * x)) if pd.notnull(x) else 0
)

# Explode the rows to have one note per row
exploded_sequences = full_solos_sequences.explode(['pitch', 'intervals', 'sixteenth_duration'])

# Ensure numeric types
exploded_sequences['intervals'] = pd.to_numeric(exploded_sequences['intervals'], errors='coerce').fillna(0).astype(float)
exploded_sequences['sixteenth_duration'] = pd.to_numeric(exploded_sequences['sixteenth_duration'], errors='coerce').fillna(0).astype(int)

# Group by melid to create sequences of tuples
grouped_sequences = exploded_sequences.groupby('melid')[['intervals', 'sixteenth_duration']].apply(
    lambda x: [(int(interval), int(duration)) for interval, duration in zip(x['intervals'], x['sixteenth_duration'])]
).reset_index(name='sequence_tuples')

# Save sequences to CSV
grouped_sequences.to_csv(output_path, index=False)
print(f"Processed sequences saved to {output_path}")
