import numpy as np
import pandas as pd
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

# Ensure that the filtered prior matrix dimensions are consistent
if filtered_prior_matrix.shape[0] != len(valid_indices):
    raise ValueError("Mismatch between filtered pattern IDs and filtered prior matrix dimensions.")


pattern_id_list = filtered_pattern_ids

# Generate initial network
def generate_initial_undirected_graph_with_weights(pattern_id_list, prior_matrix):
    graph = nx.Graph()  
    graph.add_nodes_from(pattern_id_list)

    matrix_size = prior_matrix.shape[0] 

    for i in range(matrix_size):
        valid_predecessors = [
            j for j in range(matrix_size)
            if prior_matrix[j, i] > 0
        ]
        frequency_weight = len(valid_predecessors) / matrix_size  # Normalised weight

        for predecessor in valid_predecessors:
            weight = max(1e-5, frequency_weight)  
            print(f"Adding edge between {pattern_id_list[predecessor]} and {pattern_id_list[i]} with weight {weight}")
            graph.add_edge(pattern_id_list[predecessor], pattern_id_list[i], weight=weight)

    return graph

initial_undirected_graph = generate_initial_undirected_graph_with_weights(pattern_id_list, filtered_prior_matrix)


# Save the initial graph to a csv file
output_path = os.path.join(base_dir, '../../data/output/initial_network_patterns_undirected.csv')
edges_data = [
    {'source': edge[0], 'target': edge[1], 'weight': edge[2]['weight']}
    for edge in initial_undirected_graph.edges(data=True)
]
edges_df = pd.DataFrame(edges_data)
edges_df.to_csv(output_path, index=False)

print(f"Initial undirected graph saved to {output_path}")

# Calculate Metrics

# Node metrics
metrics = {
    'node': list(initial_undirected_graph.nodes),
    'degree': [val for _, val in initial_undirected_graph.degree()],
    'betweenness_centrality': list(nx.betweenness_centrality(initial_undirected_graph).values()),
    'closeness_centrality': list(nx.closeness_centrality(initial_undirected_graph).values()),
    'pagerank': list(nx.pagerank(initial_undirected_graph).values()),
    'degree_centrality': list(nx.degree_centrality(initial_undirected_graph).values())
}

metrics_df = pd.DataFrame(metrics)

#  Output path for metrics
metrics_node_csv_path = os.path.join(base_dir, '../../data/output/node_metrics_patterns_undirected.csv')
metrics_df.to_csv(metrics_node_csv_path, index=False)
print(f"Node metrics saved to {metrics_node_csv_path}")

# Global metrics
global_metrics = {
    'number_of_nodes': [initial_undirected_graph.number_of_nodes()],
    'number_of_edges': [initial_undirected_graph.number_of_edges()],
    'average_degree': [sum(dict(initial_undirected_graph.degree()).values()) / initial_undirected_graph.number_of_nodes()],
    'density': [nx.density(initial_undirected_graph)] 
}

global_metrics_df = pd.DataFrame(global_metrics)

# Output path for metrics
metrics_global_csv_path = os.path.join(base_dir, '../../data/output/global_metrics_patterns_undirected.csv')
global_metrics_df.to_csv(metrics_global_csv_path, index=False)
print(f"Global-level metrics saved to {metrics_global_csv_path}")