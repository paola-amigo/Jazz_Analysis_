import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="network_patterns_analysis.log",
)

# Command-line arguments
parser = argparse.ArgumentParser(description="MCMC Network Analysis for Patterns")
parser.add_argument("--base_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Base directory for relative paths")
args = parser.parse_args()
base_dir = args.base_dir

# Define relative paths
output_dir = os.path.join(base_dir, "../../data/output/mcmc_results/")
node_metrics_pattern = "metrics_net_patterns_"
network_files_pattern = "final_network_patterns_"

# Load metrics for all parameter combinations
metrics_files = [file for file in os.listdir(output_dir) if file.startswith(node_metrics_pattern)]
network_files = [file for file in os.listdir(output_dir) if file.startswith(network_files_pattern)]
logging.info(f"Metrics files detected: {metrics_files}")

# Combine all metrics into a single DataFrame
metrics_dfs = []
for file in metrics_files:
    filepath = os.path.join(output_dir, file)
    params = file[len(node_metrics_pattern):-4]  # Extract parameters from the filename
    df = pd.read_csv(filepath)
    df["parameters"] = params
    metrics_dfs.append(df)

all_metrics_df = pd.concat(metrics_dfs, ignore_index=True)
logging.info(f"Shape of all_metrics_df: {all_metrics_df.shape}")

# Save combined metrics
combined_metrics_path = os.path.join(output_dir, "combined_metrics_patterns.csv")
all_metrics_df.to_csv(combined_metrics_path, index=False)
logging.info(f"Combined metrics saved to {combined_metrics_path}")

# Compute additional network metrics from final network files
network_metrics = []
for network_file in network_files:
    filepath = os.path.join(output_dir, network_file)
    params = network_file[len(network_files_pattern):-4]  # Extract parameters from the filename

    try:
        # Load the network
        edges_df = pd.read_csv(filepath)
        graph = nx.from_pandas_edgelist(edges_df, source="node_a", target="node_b", edge_attr="weight", create_using=nx.DiGraph())

        # Validate unmatched nodes
        unmatched_nodes = set(edges_df["node_a"]).union(edges_df["node_b"]) - set(graph.nodes)
        if unmatched_nodes:
            logging.warning(f"Unmatched nodes found in network: {unmatched_nodes}")

        # Compute metrics
        degree_centrality = nx.degree_centrality(graph)
        avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
        max_degree_centrality = max(degree_centrality.values())
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)

        # Add metrics to the list
        network_metrics.append({
            "parameters": params,
            "avg_degree_centrality": avg_degree_centrality,
            "max_degree_centrality": max_degree_centrality,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
        })
    except Exception as e:
        logging.error(f"Error processing network file {network_file}: {e}")

# Convert to DataFrame and merge with existing metrics
network_metrics_df = pd.DataFrame(network_metrics)
metrics_summary = all_metrics_df.groupby("parameters").agg(
    avg_log_likelihood=("log_likelihood", "mean"),
    max_log_likelihood=("log_likelihood", "max"),
    avg_network_size=("network_size", "mean"),
    max_network_size=("network_size", "max"),
    std_log_likelihood=("log_likelihood", "std"),
).reset_index()

metrics_summary = metrics_summary.merge(network_metrics_df, on="parameters", how="left")

# Save updated metrics summary
updated_summary_path = os.path.join(output_dir, "summary_metrics_patterns_updated.csv")
metrics_summary.to_csv(updated_summary_path, index=False)
logging.info(f"Updated metrics summary saved to {updated_summary_path}")

# Select the best parameters based on avg_degree_centrality
best_params = metrics_summary.sort_values(by="avg_degree_centrality", ascending=False).iloc[0]
logging.info(f"Best parameters based on avg degree centrality: {best_params['parameters']}")

# Save the best network parameters for reference as CSV
best_params_df = pd.DataFrame([best_params])
best_params_csv_path = os.path.join(output_dir, "best_network_patterns_params.csv")
best_params_df.to_csv(best_params_csv_path, index=False)
logging.info(f"Best parameters saved to {best_params_csv_path}")

# Filter the networks for visualization
filtered_params = [best_params["parameters"]]

filtered_df = all_metrics_df[all_metrics_df["parameters"].isin(filtered_params)].copy()

# Define parameter details dynamically
unique_params = all_metrics_df["parameters"].unique()
parameter_details = {
    param: f"Iterations: {param.split('_')[0]}, Burn-In: {param.split('_')[1]}, "
           f"Thinning: {param.split('_')[2]}, Threshold: {param.split('_')[3]}, "
           f"Modifier: {param.split('_')[4]}, Max Iterations: {param.split('_')[5]}"
    for param in unique_params
}

# Add descriptions for filtered best parameters
filtered_df["description"] = filtered_df["parameters"].map(parameter_details)

# Check if descriptions were successfully added
if filtered_df["description"].isnull().any():
    logging.warning("Some parameters in filtered_df do not have a description.")

# Save filtered likelihood data
filtered_likelihood_path = os.path.join(output_dir, "filtered_likelihood_data_patterns.csv")
filtered_df.to_csv(filtered_likelihood_path, index=False)
logging.info(f"Filtered likelihood data saved to {filtered_likelihood_path}")

# Plot log likelihood evolution with detailed parameter descriptions
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_df, x="iteration", y="log_likelihood", hue="description", palette="Set2")
plt.title("Log Likelihood Evolution (Filtered Parameters)")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.legend(title="Parameters", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.tight_layout()
plt.show()