# Jazz Solos Influence Analysis

This project analyses jazz solos to infer musical influences using Sequence alignment (Mongeau-Sankoff),Bayesian inference and MCMC sampling. It processes similarity and prior matrices to produce a network graph showing directional influences between performers.

---

## Prerequisites

- Python 3.9 or higher
- Dependencies listed in `requirements.txt`

---

## **Getting Started**

### **1. Download the Project Files**

Ensure all project files are downloaded into a local directory. Key files include:

- `src/` (contains scripts)
- `data/input/` (for input files)
- `data/output/` (for output files, created after the script runs)
- `requirements.txt`

---

### **2. Set Up the Environment**

#### a. Create and activate a virtual environment:

On Linux/MacOS:
python3.96 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
---

### 2. Set Up the Environment

#### a. Create and activate a virtual environment:

On Linux/MacOS:
python3.9 -m venv .venv
source .venv/bin/activate

#### b. Install dependencies

pip install -r requirements.txt

----------
## **Dataset**

The preprocessing pipeline uses the Weimar Jazz Database (WJazzD) as the primary dataset. Ensure that the `wjazzd.db` file is available in the `data/` directory.

Download the database from the [WJazzD Repository](https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db) if required.

---

### **Required Tools**

#### **Melpat**
1. **Purpose**: Extracts musical patterns from solos.
2. **Installation**:
   - Download the MeloSpy Suite and add the Melpat executable to your system PATH.
   chmod +x /path/to/melpat
3. **Configuration**:
The melpat_config.yaml file is located in the config directory. This file contains necessary parameters for running the tool. It can be adjust settings in the file as required for your analysis.
Path: config/melpat_config.yaml
Integration:
The pattern_extraction.py script automatically references the melpat_config.yaml file during execution.

Output Example:

| id                                | start | N | onset  | dur   | metricalposition | value               | freq | prob100 |
|-----------------------------------|-------|---|--------|-------|------------------|---------------------|------|---------|
| ArtPepper_Anthropology_FINAL.sv   | 0     | 5 | 10.343 | 1.02  | 4.1.0.1.1        | [65,63,58,61,63]    | 5    | 0.003   |
| ArtPepper_Anthropology_FINAL.sv   | 1     | 5 | 10.638 | 1.103 | 4.4.0.2.1        | [63,58,61,63,58]    | 9    | 0.005   |

------------

Analysis Pipeline

## **Preprocessing**
In this stage the data is extracted from the sqLITE3 database file (wjazzd.db) to create the base dataset to analyse on this project and also the external tool melpat from the Jazzomat project is used to extract patterns from the solos using the same database mentioned.

### **Run the Preprocessing scripts**

Run all preprocessing scripts in sequence:
python src/preprocessing/run_preprocessing.py

### **Preprocessing Steps**

The `run_preprocessing.py` script executes the following steps:

1. **Extract Data**:
   - `solos_db_extraction.py`: Extracts raw data from the `wjazzd.db` dataset.

2. **Calculate Durations**:
   - `calculate_durations.py`: Computes note durations for all solos.

3. **Clean Data**:
   - `solos_db_clean.py`: Cleans and filters the extracted data to remove inconsistencies.

4. **Extract Patterns**:
   - `pattern_extraction.py`: Runs Melpat to extract musical patterns using the `config/melpat_config.yaml` file.

5. **Process Patterns**:
   - `patterns_processed.py`: Processes the extracted patterns to ensure consistency.

6. **Match Patterns to Solos**:
   - `matched_patterns.py`: Matches extracted patterns to specific solos in the dataset.

7. **Clean Patterns**:
   - `patterns_clean.py`: Cleans and filters patterns for analysis.

8. **Select Top Patterns**:
   - `patterns_top_sample.py`: Extracts a representative sample of top patterns.

### **Output**

Processed data and results will be saved in the `data/output/` directory.

---

## **Alignment**

**Purpose**: Align musical sequences derived from jazz solos and patterns to measure similarity. This step uses the Mongeau-Sankoff algorithm for both solos and patterns.

---

### **Steps**

1. **Sequence Generation**:
   - Converts pattern and solo data into Mongeau-Sankoff-compatible sequences.
   - **Scripts**:
     - `MS_sequences_patterns.py`: Processes patterns.
     - `MS_sequences_solos.py`: Processes solos.
   - **Command**:
     python src/preprocessing/MS_sequences_patterns.py
     python src/preprocessing/MS_sequences_solos.py
     ```

2. **Mongeau-Sankoff Alignment**:
   - Aligns the sequences to calculate similarity scores.
   - Scripts:
     - `run_mongeau_sankoff_patterns.py`: Aligns patterns.
     - `run_mongeau_sankoff_solos.py`: Aligns solos.
   - **Command**:
     python src/alignment/run_mongeau_sankoff_patterns.py
     python src/alignment/run_mongeau_sankoff_solos.py
     ```

### **Outputs**:
- Pattern alignments: `data/output/aligned_patterns_results_k2_{parameter}.csv`
- Solo alignments: `data/output/aligned_solo_results_k2_{parameter}.csv`

The resulting files are the basis for generating similarity matrices.
## **Prior Matrices Generation**

**Purpose**: Generate matrices that encode temporal and stylistic constraints for Bayesian inference. These matrices ensure that the directionality of influence in the networks follows logical chronological and stylistic relationships.

---

### **Steps**

1. **Generate Prior Matrices for Patterns**:
   - Generates a matrix that defines constraints for pattern influences.
   - **Script**: `priors_matrix_patterns.py`
   - **Command**:
     python src/bayes/priors_matrix_patterns.py
     ```

2. **Generate Prior Matrices for Solos**:
   - Generates a matrix that defines constraints for solo influences.
   - **Script**: `priors_matrix_solos.py`
   - **Command**:
     python src/bayes/priors_matrix_solos.py
     ```

---

### **Outputs**

- **Patterns Prior Matrix**: `data/output/priors_matrix_patterns.csv`
- **Solos Prior Matrix**: `data/output/priors_matrix_solos.csv`

---

### **Description of Prior Matrices**

- **Temporal Constraints**:
  - Ensures that influences flow only from earlier to later recordings.
- **Stylistic Constraints**:
  - Adjusts influence likelihoods based on genre similarity and performer style.

These matrices serve as input for the MCMC sampling stage, refining the networks using Bayesian inference.

---

## **Initial Network Generation**

**Purpose**: Build directed network graphs for patterns and solos using similarity scores.

---

### **Steps**

1. **Generate Initial Networks**:
   - Uses alignment outputs to create directed networks.
   - **Scripts**:
     - `gen_ini_net_patterns_dir.py`: Generates a network for patterns.
     - `gen_ini_net_solos_dir.py`: Generates a network for solos.
   - **Command**:
     python src/network/generate_initial_network.py
     ```

---

## **MCMC Sampling**

**Purpose**: Use Bayesian inference and MCMC to refine the networks by incorporating prior constraints (temporal order).

---

### **Steps**

1. **Generate Prior Matrices**:
   - Enforce chronological constraints.
   - **Scripts**:
     - `priors_matrix_patterns.py`: Generates prior matrix for patterns.
     - `priors_matrix_solos.py`: Generates prior matrix for solos.
   - **Command**:
     python src/bayes/generate_prior_matrices.py

2. **Run MCMC Sampling**:
   - Refines the initial networks using Bayesian inference by incorporating the prior and similarity matrices.
   - **Scripts**:
     - `mcmc_sampling_patterns.py`
     - `mcmc_sampling_solos.py`
   - **Command**:
     python src/bayes/mcmc_sampling_patterns.py
     python src/bayes/mcmc_sampling_solos.py

---

### **Outputs**

- ** Pattern Network**: `data/output/mcmc_results/final_network_patterns_{parameters}.csv`
- **Solo Network**: `data/output/mcmc_results/final_network_solo_{parameters}.csv`

---

## **Network Analysis and Visualisation**

**Purpose**: Perform a detailed analysis of the refined networks and visualise the directional influences between patterns and solos.

---

### **Steps**

#### 1. Perform Network Analysis for Patterns and Solos
- Analyzes the refined pattern and solo networks to calculate metrics and generate adjacency lists.
- **Script**: `network_patterns_analysis.py`
- **Command**:
  python src/network/network_patterns_analysis.py

## Prior Matrices Generation

**Purpose**: Generate matrices that encode temporal and stylistic constraints for Bayesian inference. These matrices ensure that the directionality of influence in the networks follows logical chronological and stylistic relationships.

### Steps:

1. **Generate Prior Matrices for Patterns**:
   - Generates a matrix that defines constraints for pattern influences.
   - Script: `priors_matrix_patterns.py`
   - **Command**:
     python src/bayes/priors_matrix_patterns.py


2. **Generate Prior Matrices for Solos**:
   - Generates a matrix that defines constraints for solo influences.
   - Script: `priors_matrix_solos.py`
   - **Command**:
     python src/bayes/priors_matrix_solos.py

### Outputs:
- **Patterns Prior Matrix**: `data/output/priors_matrix_patterns.csv`
- **Solos Prior Matrix**: `data/output/priors_matrix_solos.csv`

### Description of Prior Matrices:
- **Temporal Constraints**:
  - Ensures that influences flow only from earlier to later recordings.
- **Stylistic Constraints**:
  - Adjusts influence likelihoods based on genre similarity and performer style.

These matrices serve as input for the MCMC sampling stage, refining the networks using Bayesian inference.

## Initial Network Generation

**Purpose**: Build directed network graphs for patterns and solos using similarity scores.

### Steps:

1. **Generate Initial Networks**:
   - Uses alignment outputs to create directed networks.
   - Scripts:
     - `gen_ini_net_patterns_dir.py`: Generates a network for patterns.
     - `gen_ini_net_solos_dir.py`: Generates a network for solos.

   **Command**:
   python src/network/generate_initial_network.py

---

#### **3. MCMC Sampling**

Include a section describing MCMC, its purpose, and its implementation.

##### **MCMC Sampling Section**

## MCMC Sampling

**Purpose**: Use Bayesian inference and MCMC to refine the networks by incorporating prior constraints (e.g., temporal order, stylistic similarities).

### Steps:

1. **Generate Prior Matrices**:
   - Enforce chronological and stylistic constraints.
   - Scripts:
     - `priors_matrix_patterns.py`: Generates prior matrix for patterns.
     - `priors_matrix_solos.py`: Generates prior matrix for solos.
   - **Command**:
     python src/bayes/generate_prior_matrices.py


2. **Run MCMC Sampling**:
   - Refines the initial networks using Bayesian inference passing the prior matrices and similarity matrices
   - Scripts:
     - `mcmc_sampling_patterns.py`
     - `mcmc_sampling_solos.py`
   - **Command**:
     python src/bayes/mcmc_sampling_patterns.py
     python src/bayes/mcmc_sampling_solos.py


3. **Output**:
   - Refined pattern network: `data/output/mcmc_results/pattern_network_refined.csv`
   - Refined solo network: `data/output/mcmc_results/solo_network_refined.csv`

## **Network Analysis and Visualisation**

**Purpose**: Perform a detailed analysis of the derived networks and visualise the directional influences between patterns and solos.

---

### **Steps**

#### 1. Perform Network Analysis for Patterns and Solos

- Analyzes the pattern networks and calculate metrics
- **Script**: `network_patterns_analysis.py`
- **Command**:
  python src/network/network_patterns_analysis.py

- Display the pattern networks and metrics
- **Command**:
  python src/network/network_patterns.py
 - Analyzes solo networks and calculate metrics
- **Script**: `network_solo_analysis.py`

- **Command**:
  python src/network/network_solo_analysis.py
- Display the solo networks and metrics
- **Command**:
  python src/network/network_solos.py



