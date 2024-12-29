import os
import numpy as np
import pandas as pd
import random
import logging
import networkx as nx
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='mcmc_sampling.log'
)

# Define base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load prior matrix and similarity matrix
prior_matrix_path = os.path.join(base_dir, '../../data/output/prior_matrix_solos.npy')
similarity_matrix_path = os.path.join(base_dir, '../../data/output/similarity_matrix_solos_k2_0.75.csv')

prior_matrix = np.load(prior_matrix_path)
similarity_matrix_df = pd.read_csv(similarity_matrix_path, index_col=0)
similarity_matrix_df.columns = similarity_matrix_df.columns.astype(int)
print(f"Prior matrix shape: {prior_matrix.shape}")
print(f"Similarity matrix shape: {similarity_matrix_df.shape}")
print("First 10 indices in similarity matrix:")
print(similarity_matrix_df.index[:10])
print("First 10 columns in similarity matrix:")
print(similarity_matrix_df.columns[:10])
print(f"Data type of similarity matrix indices: {similarity_matrix_df.index.dtype}")
print(f"Data type of similarity matrix columns: {similarity_matrix_df.columns.dtype}")
# Validate index and columns of similarity_matrix_df are integers
similarity_matrix_df.index = similarity_matrix_df.index.astype(int)
similarity_matrix_df.columns = similarity_matrix_df.columns.astype(int)

# Extract metadata to align `melid` with prior matrix
metadata_path = os.path.join(base_dir, '../../data/solos_db_with_durations_cleaned.csv')
metadata_df = pd.read_csv(metadata_path)

# Subset metadata to include only necessary columns
metadata_subset = metadata_df[['melid', 'title', 'performer']].drop_duplicates()

# Check for consistency between prior matrix and metadata
melid_list = metadata_subset['melid'].tolist()
if prior_matrix.shape[0] != len(melid_list):
    raise ValueError("Mismatch between prior matrix dimensions and `melid` list length.")

melid_to_index = {melid: idx for idx, melid in enumerate(melid_list)}
index_to_melid = {idx: melid for melid, idx in melid_to_index.items()}

# Filter similarity matrix and prior matrix based on common IDs
common_ids = sorted(set(similarity_matrix_df.index) & set(melid_list))
similarity_matrix = similarity_matrix_df.loc[common_ids, common_ids].values
prior_matrix = prior_matrix[np.ix_(
    [melid_to_index[melid] for melid in common_ids],
    [melid_to_index[melid] for melid in common_ids]
)]

# Define output path
output_dir = os.path.join(base_dir, '../../data/output/mcmc_results/')
os.makedirs(output_dir, exist_ok=True)

# Function to calculate log-likelihood
def calculate_log_likelihood(network_edges, similarity_matrix, prior_matrix):
    total_log_likelihood = 0.0
    epsilon = 1e-10
    for (node_a, node_b, _) in network_edges:
        index_a = melid_to_index[node_a]
        index_b = melid_to_index[node_b]
        similarity_score = similarity_matrix[index_a, index_b]
        prior_score = prior_matrix[index_a, index_b]
        total_log_likelihood += np.log(similarity_score + epsilon) + np.log(prior_score + epsilon)
    return total_log_likelihood

# MCMC sampling function
def mcmc_sampler(
    initial_network, similarity_matrix, prior_matrix,
    iterations=1000, burn_in=500, thinning=10,
    threshold=0.3, acceptance_modifier=0.1, edge_search_iterations=50
):
    current_network = initial_network
    current_log_likelihood = calculate_log_likelihood(current_network, similarity_matrix, prior_matrix)
    sampled_networks = []
    metrics = []

    for i in range(iterations):
        proposed_network = current_network[:]
        if random.uniform(0, 1) > 0.5 and proposed_network:
            # Modify edge
            edge_to_modify = random.choice(proposed_network)
            proposed_network.remove(edge_to_modify)
            new_weight = similarity_matrix[
                melid_to_index[edge_to_modify[0]], melid_to_index[edge_to_modify[1]]
            ] * (1 + random.uniform(-acceptance_modifier, acceptance_modifier))
            proposed_network.append((edge_to_modify[0], edge_to_modify[1], new_weight))
        else:
            # Add edge
            for _ in range(edge_search_iterations):
                node_a, node_b = random.sample(common_ids, 2)
                if node_a != node_b and similarity_matrix[melid_to_index[node_a], melid_to_index[node_b]] > threshold:
                    proposed_network.append((
                        node_a, node_b, similarity_matrix[melid_to_index[node_a], melid_to_index[node_b]]
                    ))
                    break

        proposed_log_likelihood = calculate_log_likelihood(proposed_network, similarity_matrix, prior_matrix)
        delta_likelihood = proposed_log_likelihood - current_log_likelihood
        acceptance_ratio = min(1, np.exp(delta_likelihood))

        if random.uniform(0, 1) < acceptance_ratio:
            current_network = proposed_network
            current_log_likelihood = proposed_log_likelihood

        if i >= burn_in and (i - burn_in) % thinning == 0:
            sampled_networks.append(current_network)

        metrics.append({
            "iteration": i,
            "log_likelihood": current_log_likelihood,
            "network_size": len(current_network)
        })

    return sampled_networks[-1] if sampled_networks else current_network, metrics

# Define parameter grid
parameter_grid = {
    "iterations": [1000, 5000],
    "burn_in": [500, 2500],
    "thinning": [1, 5],
    "threshold": [0.2, 0.3],
    "acceptance_modifier": [0.1, 0.2],
    "edge_search_iterations": [50, 100]
}

# Generate all parameter combinations
experiments = list(itertools.product(
    parameter_grid["iterations"],
    parameter_grid["burn_in"],
    parameter_grid["thinning"],
    parameter_grid["threshold"],
    parameter_grid["acceptance_modifier"],
    parameter_grid["edge_search_iterations"]
))

# Run experiments
for params in experiments:
    iterations, burn_in, thinning, threshold, acceptance_modifier, edge_search_iterations = params
    logging.info(f"Running experiment with parameters: {params}")

    try:
        final_network, metrics = mcmc_sampler(
            [], similarity_matrix, prior_matrix,
            iterations=iterations, burn_in=burn_in, thinning=thinning,
            threshold=threshold, acceptance_modifier=acceptance_modifier, edge_search_iterations=edge_search_iterations
        )

        # Save results
        param_str = f"{iterations}_{burn_in}_{thinning}_{threshold}_{acceptance_modifier}_{edge_search_iterations}"
        final_network_file = os.path.join(output_dir, f'final_network_solos_{param_str}.csv')
        final_network_df = pd.DataFrame(final_network, columns=['node_a', 'node_b', 'weight'])
        final_network_df.to_csv(final_network_file, index=False)

        metrics_file = os.path.join(output_dir, f'metrics_net_solos_{param_str}.csv')
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(metrics_file, index=False)

        logging.info(f"Results saved for parameters: {param_str}")
    except Exception as e:
        logging.error(f"Error for parameters {params}: {e}")