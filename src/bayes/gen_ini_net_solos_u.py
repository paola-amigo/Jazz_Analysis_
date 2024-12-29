import numpy as np
import pandas as pd
import random
import networkx as nx
import os

# Define relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))
prior_matrix_path = os.path.join(base_dir, '../../data/output/prior_matrix_solos.npy')
output_csv_path = os.path.join(base_dir, '../../data/output/initial_network_solos_undirected.csv')
metrics_node_csv_path = os.path.join(base_dir, '../../data/output/initial_network_metrics_nodes_undirected.csv')
metrics_global_csv_path = os.path.join(base_dir, '../../data/output/initial_network_metrics_global_undirected.csv')

# Load prior matrix
prior_matrix_solos = np.load(prior_matrix_path)

# Extract and clean solo IDs, title, and performer from the metadata
metadata_df = pd.read_csv(os.path.join(base_dir, '../../data/solos_db_with_durations_cleaned.csv'))

# Subset the metadata to only include melid, title, and performer
metadata_subset = metadata_df[['melid', 'title', 'performer']]

# Drop duplicates based on melid
metadata_cleaned = metadata_subset.drop_duplicates(subset=['melid'])

# Ensure no duplicates exist for melid
if metadata_cleaned['melid'].duplicated().any():
    raise ValueError("Duplicate melid entries found after cleaning!")

# Print the cleaned metadata
print(metadata_cleaned.head())

# Use the cleaned melid list for graph construction
solo_id_list = metadata_cleaned['melid'].tolist()

# Function to generate an initial undirected network
def generate_initial_network_undirected(solo_id_list, prior_matrix, random_selection=True, seed_value=42):
    """
    Generates an initial undirected network using a prior matrix.
    
    Parameters:
        solo_id_list (list): List of solo IDs corresponding to the prior matrix dimensions.
        prior_matrix (np.ndarray): Prior matrix used to define edge weights.
        random_selection (bool): If True, selects a random predecessor. Otherwise, uses the highest value.
        seed_value (int): Seed for reproducibility of random operations.

    Returns:
        nx.Graph: Generated undirected network.
    """
    # Set the random seed for reproducibility
    random.seed(seed_value)

    # Create an empty undirected graph and add all nodes
    network = nx.Graph()
    network.add_nodes_from(solo_id_list)

    matrix_size = prior_matrix.shape[0]

    for i in range(matrix_size):
        # Find valid predecessors (those with a value of 1 in the prior matrix)
        valid_predecessors = [j for j in range(matrix_size) if prior_matrix[j, i] == 1]

        # If there are no valid predecessors, skip the current node
        if not valid_predecessors:
            continue

        # Choose a predecessor
        if random_selection:
            predecessor = random.choice(valid_predecessors)
        else:
            predecessor = max(valid_predecessors, key=lambda x: prior_matrix[x, i])

        # Add an edge with a weight of 1 (or any meaningful value you prefer)
        weight = 1
        network.add_edge(solo_id_list[predecessor], solo_id_list[i], weight=weight)

        print(f"Adding edge between {solo_id_list[predecessor]} and {solo_id_list[i]} with weight {weight}")

    return network

# Generate the initial undirected network
initial_network = generate_initial_network_undirected(solo_id_list, prior_matrix_solos)

# Add final validation step before saving
for node in initial_network.nodes():
    if node not in solo_id_list:
        raise ValueError(f"Node {node} is not a valid solo ID. Check the prior matrix dimensions.")

# Convert the network edges to a DataFrame
edges_data = [
    {'source': edge[0], 'target': edge[1], 'weight': edge[2]['weight']}
    for edge in initial_network.edges(data=True)
]
edges_df = pd.DataFrame(edges_data)

# Save the edges as a CSV file
edges_df.to_csv(output_csv_path, index=False)
print(f"Initial undirected network saved to {output_csv_path}")

# --- Compute Metrics ---

# Node-level metrics
metrics = {
    'node': list(initial_network.nodes),
    'degree': [val for _, val in initial_network.degree()],
    'betweenness_centrality': list(nx.betweenness_centrality(initial_network).values()),
    'closeness_centrality': list(nx.closeness_centrality(initial_network).values()),
    'pagerank': list(nx.pagerank(initial_network).values()),
    'degree_centrality': list(nx.degree_centrality(initial_network).values())
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(metrics_node_csv_path, index=False)
print(f"Node-level metrics saved to {metrics_node_csv_path}")

# Global-level metrics
global_metrics = {
    'number_of_nodes': [initial_network.number_of_nodes()],
    'number_of_edges': [initial_network.number_of_edges()],
    'average_degree': [sum(dict(initial_network.degree()).values()) / initial_network.number_of_nodes()],
    'density': [nx.density(initial_network)]
}

global_metrics_df = pd.DataFrame(global_metrics)

# Save global-level metrics
global_metrics_df.to_csv(metrics_global_csv_path, index=False)
print(f"Global-level metrics saved to {metrics_global_csv_path}")