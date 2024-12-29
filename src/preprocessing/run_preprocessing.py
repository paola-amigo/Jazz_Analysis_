import subprocess
import os

# Define the base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# List of scripts to execute in order
scripts = [
    "src/preprocessing/solos_db_extraction.py",
    "src/preprocessing/calculate_durations.py",
    "src/preprocessing/solos_db_clean.py",
    "src/preprocessing/pattern_extraction.py",
    "src/preprocessing/patterns_processed.py",
    "src/preprocessing/matched_patterns.py",
    "src/preprocessing/patterns_clean.py",
    "src/preprocessing/patterns_top_sample.py"
]

# Execute each script in order
for script in scripts:
    script_path = os.path.join(BASE_DIR, script)
    print(f"Running {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error occurred in {script_path}:\n{result.stderr}")
        break
    else:
        print(f"Finished {script_path}:\n{result.stdout}")

print("Preprocessing completed successfully!")