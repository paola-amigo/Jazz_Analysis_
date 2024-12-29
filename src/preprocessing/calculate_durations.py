import pandas as pd
import os

# Define relative paths for input and output
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')
input_file = os.path.join(data_dir, 'solos_db.csv')
output_file = os.path.join(data_dir, 'solos_db_with_durations.csv')

# Load the DataFrame from the CSV file (solos_db)
solos_db = pd.read_csv(input_file)

# Updated duration calculation function
def calculate_duration_with_tatum(row, next_row):
    # Base durations for each division type
    division_mapping = {
        1: 1 / 4,    # Quarter note
        2: 1 / 8,    # Eighth note
        3: 1 / 8,    # Approximate to eighth note for consistency with sixteenths
        4: 1 / 16,   # Sixteenth note
    }

    # For divisions 4 and above, default to sixteenth note
    if row['division'] >= 4:
        base_duration = 1 / 16
    else:
        # Get the base duration from the division mapping if it exists
        base_duration = division_mapping.get(row['division'], None)

    if base_duration is None:
        # If no valid base duration is found, return None
        return None

    # Calculate the duration based on the tatum positions
    if (
        next_row is not None and 
        row['bar'] == next_row['bar'] and
        row['beat'] == next_row['beat']
    ):
        tatum_current = row['tatum']
        tatum_next = next_row['tatum']
        tatum_difference = tatum_next - tatum_current
    else:
        # If it's the last note in the beat or the last note of the sequence
        tatum_difference = 1  # Assume it occupies a single tatum if it's the last one or no subsequent notes

    # Multiply base duration by tatum difference
    return base_duration * tatum_difference

# Function to convert durations into sixteenth-note representation
def convert_to_sixteenth_representation(duration):
    # Convert to sixteenth representation: 1/4 becomes 4, 1/8 becomes 2, 1/16 becomes 1
    if duration == 1 / 4:
        return 4
    elif duration == 1 / 8:
        return 2
    elif duration == 1 / 16:
        return 1
    else:
        # For other durations, calculate the equivalent "divided by 1/16"
        return int(duration / (1 / 16))

# Apply the function to the DataFrame
durations = []
sixteenth_representations = []

for i in range(len(solos_db)):
    current_row = solos_db.iloc[i]
    next_row = solos_db.iloc[i + 1] if i + 1 < len(solos_db) else None
    duration = calculate_duration_with_tatum(current_row, next_row)
    durations.append(duration)
    sixteenth_representations.append(convert_to_sixteenth_representation(duration) if duration is not None else None)

# Assign calculated durations back to the DataFrame
solos_db['calculated_duration'] = durations
solos_db['sixteenth_representation'] = sixteenth_representations

# Verify the output
print(solos_db[['bar', 'beat', 'tatum', 'division', 'calculated_duration', 'sixteenth_representation']].head(20))

# Export the updated DataFrame to a CSV file
solos_db.to_csv(output_file, index=False)
print("Durations added and saved to solos_db_with_durations.csv")