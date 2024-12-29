import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import argparse

# Command-line arguments
parser = argparse.ArgumentParser(description="MCMC Network Analysis")
parser.add_argument("--base_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Base directory for relative paths")
args = parser.parse_args()
base_dir = args.base_dir

# Define relative paths
output_dir = os.path.join(base_dir, '../../data/output/mcmc_results/')
node_metrics_pattern = 'metrics_net_solos_'
network_files_pattern = 'final_network_solos_'

# Load metrics for all parameter combinations
metrics_files = [file for file in os.listdir(output_dir) if file.startswith(node_metrics_pattern)]
network_files = [file for file in os.listdir(output_dir) if file.startswith(network_files_pattern)]
print("Metrics files detected:", metrics_files)
# Combine all metrics into a single DataFrame
metrics_dfs = []
for file in metrics_files:
    filepath = os.path.join(output_dir, file)
    params = file[len(node_metrics_pattern):-4]  # Extract parameters from the filename
    df = pd.read_csv(filepath)
    df['parameters'] = params
    metrics_dfs.append(df)

all_metrics_df = pd.concat(metrics_dfs, ignore_index=True)
print(f"Shape of all_metrics_df: {all_metrics_df.shape}")

# Save combined metrics
combined_metrics_path = os.path.join(output_dir, 'combined_metrics_solos.csv')
all_metrics_df.to_csv(combined_metrics_path, index=False)
print(f"Combined metrics saved to {combined_metrics_path}")

# Compute additional network metrics from final network files
network_metrics = []

for network_file in network_files:
    filepath = os.path.join(output_dir, network_file)
    params = network_file[len(network_files_pattern):-4]  # Extract parameters from the filename
    
    try:
        # Load the network
        edges_df = pd.read_csv(filepath)
        graph = nx.from_pandas_edgelist(edges_df, source='node_a', target='node_b', edge_attr='weight', create_using=nx.DiGraph())
        
        # Compute metrics
        degree_centrality = nx.degree_centrality(graph)
        avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
        max_degree_centrality = max(degree_centrality.values())
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        
        # Append metrics to the list
        network_metrics.append({
            'parameters': params,
            'avg_degree_centrality': avg_degree_centrality,
            'max_degree_centrality': max_degree_centrality,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density
        })
    except Exception as e:
        print(f"Error processing network file {network_file}: {e}")

# Convert to DataFrame and merge with existing metrics
network_metrics_df = pd.DataFrame(network_metrics)
metrics_summary = all_metrics_df.groupby('parameters').agg(
    avg_log_likelihood=('log_likelihood', 'mean'),
    max_log_likelihood=('log_likelihood', 'max'),
    avg_network_size=('network_size', 'mean'),
    max_network_size=('network_size', 'max'),
    std_log_likelihood=('log_likelihood', 'std')
).reset_index()

metrics_summary = metrics_summary.merge(network_metrics_df, on='parameters', how='left')

# Save updated metrics summary
updated_summary_path = os.path.join(output_dir, 'summary_metrics_solos_updated.csv')
metrics_summary.to_csv(updated_summary_path, index=False)
print(f"Updated metrics summary saved to {updated_summary_path}")

# Create a mapping for parameter details
parameter_details = {
    "1000_500_10_0.7_0.2_100": "Burn-In: 1000, Interval: 500, Scaling: 0.7, Acceptance: 0.2, Max Iterations: 100",
    "1000_500_1_0.7_0.2_10": "Burn-In: 1000, Interval: 500, Scaling: 0.7, Acceptance: 0.2, Max Iterations: 10",
    # Add descriptions for other parameters as necessary
}

# Select the best parameters based on avg_degree_centrality
best_params = metrics_summary.sort_values(by='avg_degree_centrality', ascending=False).iloc[0]
print(f"Best parameters based on avg degree centrality: {best_params['parameters']}")

# Save the best network parameters for reference as CSV
best_params_df = pd.DataFrame([best_params])
best_params_csv_path = os.path.join(output_dir, 'best_network_solos_params.csv')
best_params_df.to_csv(best_params_csv_path, index=False)
print(f"Best parameters saved to {best_params_csv_path}")


# Filter the networks for visualization
filtered_params = [best_params['parameters']]  # Only the best network

# Add descriptions for filtered parameters
filtered_df = all_metrics_df[all_metrics_df['parameters'].isin(filtered_params)].copy()
filtered_df['description'] = filtered_df['parameters'].map(parameter_details)

# Save filtered likelihood data
filtered_likelihood_path = os.path.join(output_dir, 'filtered_likelihood_data.csv')
filtered_df.to_csv(filtered_likelihood_path, index=False)
print(f"Filtered likelihood data saved to {filtered_likelihood_path}")

# Plot log likelihood evolution with detailed parameter descriptions
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_df, x='iteration', y='log_likelihood', hue='description', palette='Set2')
plt.title('Log Likelihood Evolution (Filtered Parameters)')
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()