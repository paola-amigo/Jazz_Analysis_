import os
import subprocess
import pandas as pd

# Set the base directory to the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the input file
input_file_path = os.path.join(BASE_DIR, '..', 'data', 'solos_db_with_durations.csv')

# Validate input file
if not os.path.exists(input_file_path):
    print(f"Error: Input file does not exist at {input_file_path}")
    exit(1)

# Load the DataFrame from the CSV file
try:
    solos_full = pd.read_csv(input_file_path)
    print("First 10 rows of input file:")
    print(solos_full.head(10))
except Exception as e:
    print(f"Error reading input file: {e}")
    exit(1)

# Paths for the melpat executable and configuration file
melpat_path = '/Users/paola_amigo/Desktop/Thesis/melospy-suite_V_1_6_mac_osx/bin/melpat'
config_path = '/Users/paola_amigo/Desktop/Thesis/JazzSolos/melpat_config.yaml'

# Validate melpat executable
if not os.path.exists(melpat_path):
    print(f"Error: Melpat executable does not exist at {melpat_path}")
    exit(1)

# Validate configuration file
if not os.path.exists(config_path):
    print(f"Error: Configuration file does not exist at {config_path}")
    exit(1)

# Run the melpat command
try:
    command = [
        melpat_path,
        '-c', config_path,  # Specify the configuration file
        '--verbose'
    ]

    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    # Log success
    print("Pattern extraction successful!")
    print("Output from melpat:")
    print(result.stdout)

    # Optionally, write output to a log file
    output_log_path = os.path.join(BASE_DIR, '..', 'output', 'melpat_output.log')
    with open(output_log_path, 'w') as log_file:
        log_file.write(result.stdout)

except subprocess.CalledProcessError as e:
    print(f"Error running melpat: {e}")
    print("Command output:")
    print(e.stderr)