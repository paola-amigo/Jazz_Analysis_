import numpy as np
import pandas as pd
import random
import networkx as nx
import os

# Define directory + Input/Output paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load prior matrix
prior_matrix_path = os.path.join(base_dir, '../../data/output/prior_matrix_patterns.npy')
prior_matrix_patterns = np.load(prior_matrix_path)

# Load filtered top sample pattern ids from csv
top_patterns_sample_path = os.path.join(base_dir, '../../data/top_patterns_sample.csv')
top_patterns_sample = pd.read_csv(top_patterns_sample_path)
filtered_pattern_ids = top_patterns_sample['pattern_id'].unique()  

# Match filtered pattern ids to the prior matrix dimensions
pattern_id_mapping = {pattern_id: idx for idx, pattern_id in enumerate(filtered_pattern_ids)}

# Filter the prior matrix to match the filtered pattern ids
valid_indices = [pattern_id_mapping.get(pattern_id) for pattern_id in filtered_pattern_ids if pattern_id in pattern_id_mapping]
valid_indices = [idx for idx in valid_indices if idx is not None and idx < prior_matrix_patterns.shape[0]]

# Create filtered prior matrix
filtered_prior_matrix = prior_matrix_patterns[np.ix_(valid_indices, valid_indices)]

# Validate prior matrix dimensions are consistent
if filtered_prior_matrix.shape[0] != len(valid_indices):
    raise ValueError("Mismatch between filtered pattern IDs and filtered prior matrix dimensions.")

# Generate initial network
pattern_id_list = filtered_pattern_ids

# Function to generate initial network
def generate_initial_network_with_weights(pattern_id_list, prior_matrix):
    network = nx.DiGraph()
    network.add_nodes_from(pattern_id_list)
    matrix_size = prior_matrix.shape[0] 

    for i in range(matrix_size):
        valid_predecessors = [
            j for j in range(matrix_size)
            if prior_matrix[j, i] > 0
        ]
        frequency_weight = len(valid_predecessors) / matrix_size  # Normalised weight

        if valid_predecessors:
            predecessor = random.choice(valid_predecessors)
            weight = max(1e-5, frequency_weight)  # Control to keep weights as positive
            print(f"Adding edge from {pattern_id_list[predecessor]} to {pattern_id_list[i]} with weight {weight}")
            network.add_edge(pattern_id_list[predecessor], pattern_id_list[i], weight=weight)

    return network

# Generate initial network
initial_network = generate_initial_network_with_weights(pattern_id_list, filtered_prior_matrix)

# Save the initial network to a csv file
output_path = os.path.join(base_dir, '../../data/output/initial_network_patterns_directed.csv')
edges_data = [{'source': edge[0], 'target': edge[1], 'weight': edge[2]['weight']} for edge in initial_network.edges(data=True)]
edges_df = pd.DataFrame(edges_data)
edges_df.to_csv(output_path, index=False)
print(f"Initial network saved to {output_path}")

# Calculate Metrics

# Node metrics
metrics = {
    'node': list(initial_network.nodes),
    'in_degree': [val for _, val in initial_network.in_degree()],
    'out_degree': [val for _, val in initial_network.out_degree()],
    'betweenness_centrality': list(nx.betweenness_centrality(initial_network).values()),
    'closeness_centrality': list(nx.closeness_centrality(initial_network).values()),
    'pagerank': list(nx.pagerank(initial_network).values()),
    'degree_centrality': list(nx.degree_centrality(initial_network).values())
}

metrics_df = pd.DataFrame(metrics)

#  Output path for metrics
metrics_node_csv_path = os.path.join(base_dir, '../../data/output/node_metrics_patterns_directed.csv')
metrics_df.to_csv(metrics_node_csv_path, index=False)
print(f"Node-level metrics saved to {metrics_node_csv_path}")

# Global metrics
global_metrics = {
    'number_of_nodes': [initial_network.number_of_nodes()],
    'number_of_edges': [initial_network.number_of_edges()],
    'average_degree': [sum(dict(initial_network.degree()).values()) / initial_network.number_of_nodes()],
    'density': [nx.density(initial_network)]
}

global_metrics_df = pd.DataFrame(global_metrics)

# Output path for metrics
metrics_global_csv_path = os.path.join(base_dir, '../../data/output/global_metrics_patterns_directed.csv')
global_metrics_df.to_csv(metrics_global_csv_path, index=False)
print(f"Global-level metrics saved to {metrics_global_csv_path}")