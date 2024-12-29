import os
import numpy as np
import pandas as pd
import random
import logging
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='mcmc_sampling_patterns.log'
)

# Define base directory for relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load prior matrix and similarity matrix
prior_matrix_path = os.path.join(base_dir, '../../data/output/prior_matrix_patterns.npy')
similarity_matrix_path = os.path.join(base_dir, '../../data/output/similarity_matrix_patterns_k2_0.75.csv')
top_patterns_sample_path = os.path.join(base_dir, '../../data/top_patterns_sample.csv')
initial_network_path = os.path.join(base_dir, '../../data/output/initial_network_patterns_directed.csv')

prior_matrix = np.load(prior_matrix_path)
similarity_matrix_df = pd.read_csv(similarity_matrix_path, index_col=0)
top_patterns_sample = pd.read_csv(top_patterns_sample_path)

# Convert pattern IDs to integers for consistency
valid_patterns = set(top_patterns_sample['pattern_id'].astype(int))
similarity_matrix_df.index = similarity_matrix_df.index.astype(int)
similarity_matrix_df.columns = similarity_matrix_df.columns.astype(int)

# Filter the similarity matrix for valid patterns
filtered_similarity_matrix = similarity_matrix_df.loc[
    list(valid_patterns), list(valid_patterns)
].values

# Filter the prior matrix similarly
filtered_indices = [
    i for i, pattern_id in enumerate(similarity_matrix_df.index) if pattern_id in valid_patterns
]
filtered_prior_matrix = prior_matrix[np.ix_(filtered_indices, filtered_indices)]

# Update pattern ID mappings
pattern_id_list = list(valid_patterns)
pattern_to_index = {pattern_id: idx for idx, pattern_id in enumerate(pattern_id_list)}

# Load initial network from CSV
initial_network_df = pd.read_csv(initial_network_path)

# Map the edges into the expected format [(source, target, weight)]
initial_network = [
    (row['source'], row['target'], row['weight']) 
    for _, row in initial_network_df.iterrows()
]

# Validate that all nodes in the initial network exist in the pattern_id_list
valid_initial_network = [
    (source, target, weight)
    for source, target, weight in initial_network
    if source in pattern_id_list and target in pattern_id_list
]

# Define output path
output_dir = os.path.join(base_dir, '../../data/output/mcmc_results/')
os.makedirs(output_dir, exist_ok=True)

# Function to calculate log-likelihood
def calculate_log_likelihood(network_edges, similarity_matrix, prior_matrix):
    total_log_likelihood = 0.0
    epsilon = 1e-10
    for (node_a, node_b, _) in network_edges:
        if node_a not in pattern_to_index or node_b not in pattern_to_index:
            continue
        index_a = pattern_to_index[node_a]
        index_b = pattern_to_index[node_b]
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
        
        # Modify or add edges
        if random.uniform(0, 1) > 0.5 and proposed_network:
            # Modify edge
            edge_to_modify = random.choice(proposed_network)
            proposed_network.remove(edge_to_modify)
            new_weight = similarity_matrix[
                pattern_to_index[edge_to_modify[0]], pattern_to_index[edge_to_modify[1]]
            ] * (1 + random.uniform(-acceptance_modifier, acceptance_modifier))
            proposed_network.append((edge_to_modify[0], edge_to_modify[1], new_weight))
        else:
            # Add edge
            for _ in range(edge_search_iterations):
                node_a, node_b = random.sample(pattern_id_list, 2)
                if node_a != node_b and similarity_matrix[pattern_to_index[node_a], pattern_to_index[node_b]] > threshold:
                    proposed_network.append((
                        node_a, node_b, similarity_matrix[pattern_to_index[node_a], pattern_to_index[node_b]]
                    ))
                    break

        # Post-validation: Ensure no invalid edges are included
        proposed_network = [
            (source, target, weight)
            for source, target, weight in proposed_network
            if source in pattern_id_list and target in pattern_id_list
        ]

        # Calculate likelihood
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
            valid_initial_network,  # Pass the loaded and validated initial network
            filtered_similarity_matrix,
            filtered_prior_matrix,
            iterations=iterations, burn_in=burn_in, thinning=thinning,
            threshold=threshold, acceptance_modifier=acceptance_modifier, edge_search_iterations=edge_search_iterations
        )

        # Save results
        param_str = f"{iterations}_{burn_in}_{thinning}_{threshold}_{acceptance_modifier}_{edge_search_iterations}"
        final_network_file = os.path.join(output_dir, f'final_network_patterns_{param_str}.csv')
        final_network_df = pd.DataFrame(final_network, columns=['node_a', 'node_b', 'weight'])
        final_network_df.to_csv(final_network_file, index=False)

        metrics_file = os.path.join(output_dir, f'metrics_net_patterns_{param_str}.csv')
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(metrics_file, index=False)

        logging.info(f"Results saved for parameters: {param_str}")
    except Exception as e:
        logging.error(f"Error for parameters {params}: {e}")