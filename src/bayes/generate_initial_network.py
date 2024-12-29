import os
import subprocess

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(
            ['python', os.path.join('src', 'network', script_name)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}:\n{e.stderr}")
        raise

def main():
    print("Starting initial network generation...")

    # Step 1: Generate initial network for patterns
    run_script('gen_ini_net_patterns_dir.py')

    # Step 2: Generate initial network for solos
    run_script('gen_ini_net_solos_dir.py')

    print("Initial networks for patterns and solos have been successfully generated!")

if __name__ == "__main__":
    main()