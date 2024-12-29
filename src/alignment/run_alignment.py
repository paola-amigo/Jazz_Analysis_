import subprocess

# Step 1: Prepare sequences
print("Preparing Mongeau-Sankoff compatible sequences...")
subprocess.run(["python", "src/preprocessing/MS_sequences_patterns.py"], check=True)
subprocess.run(["python", "src/preprocessing/MS_sequences_solos.py"], check=True)

# Step 2: Perform alignment
print("Running Mongeau-Sankoff alignment for patterns...")
subprocess.run(["python", "src/alignment/run_mongeau_sankoff_patterns.py"], check=True)

print("Running Mongeau-Sankoff alignment for solos...")
subprocess.run(["python", "src/alignment/run_mongeau_sankoff_solos.py"], check=True)

print("Alignment process completed!")