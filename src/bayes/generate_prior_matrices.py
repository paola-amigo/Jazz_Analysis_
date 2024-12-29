import os
import subprocess

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(
            ['python', os.path.join('src', 'bayes', script_name)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}:\n{e.stderr}")
        raise

def main():
    print("Starting generation of prior matrices...")

    # Step 1: Generate prior Mmtrix for patterns
    run_script('priors_matrix_patterns.py')

    # Step 2: Generate prior matrix for solos
    run_script('priors_matrix_solos.py')

    print("All prior matrices have been successfully generated!")

if __name__ == "__main__":
    main()