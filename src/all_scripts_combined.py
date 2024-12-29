# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/network/network_solos.py
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, '../../data/output/mcmc_results/')
best_network_params_file = 'best_network_solos_params.csv'
best_network_params_path = os.path.join(output_dir, best_network_params_file)

# Load the best network parameters
if not os.path.exists(best_network_params_path):
    raise FileNotFoundError(f"Metrics file {best_network_params_path} not found.")

best_params_df = pd.read_csv(best_network_params_path)
if best_params_df.empty:
    raise ValueError("The best network parameters file is empty.")

# Extract the parameters for the best network
best_parameters = best_params_df.iloc[0, 0] 
print(f"Selected parameters for the best network: {best_parameters}")

# Look for filename for the corresponding network
final_network_file = f"final_network_solos_{best_parameters}.csv"
final_network_path = os.path.join(output_dir, final_network_file)

# Load the final network data
edges_df = pd.read_csv(final_network_path)
graph = nx.from_pandas_edgelist(edges_df, source='node_a', target='node_b', edge_attr='weight', create_using=nx.DiGraph())

# Metrics for selected network
degree_centrality = nx.degree_centrality(graph)
sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
top_10_nodes = [node for node, centrality in sorted_degree[:10]]

# Load solos dataframe for labels
solos_db_path = os.path.join(base_dir, '../../data/solos_db_with_durations_cleaned.csv')
solos_db_df = pd.read_csv(solos_db_path)
solos_db_subset = solos_db_df[['melid', 'title', 'performer']].drop_duplicates(subset=['melid'])
if solos_db_subset['melid'].duplicated().any():
    raise ValueError("Duplicate melid")
melid_to_labels = solos_db_subset.set_index('melid')[['title', 'performer']].to_dict('index')

# Plot 1: Directed graph of solos connections
layout = nx.spring_layout(graph, seed=42, k=2)
plt.figure(figsize=(14, 12))
node_sizes = [3000 * degree_centrality[node] for node in graph.nodes()]
node_colors = ['#1f78b4' if node not in top_10_nodes else '#f97c7c' for node in graph.nodes()]
node_labels = {
    node: f"{melid_to_labels[node]['title']} ({melid_to_labels[node]['performer']})" 
    if node in melid_to_labels else str(node) for node in graph.nodes()
}

nx.draw(
    graph,
    pos=layout,
    labels=node_labels,
    with_labels=True,
    node_size=node_sizes,
    node_color=node_colors,
    edge_color='gray',
    alpha=0.8,
    font_size=8,
    font_color='darkblue'
)
plt.title(f"Full Network Visualization\nTop Nodes Highlighted", fontsize=14)
plt.tight_layout()
plt.show()

# Find top 10 nodes
top_10_data = [{'Node': node, 'Title': melid_to_labels[node]['title'] if node in melid_to_labels else "Unknown", 
                'Performer': melid_to_labels[node]['performer'] if node in melid_to_labels else "Unknown",
                'Degree Centrality': degree_centrality[node]} for node in top_10_nodes]

top_10_df = pd.DataFrame(top_10_data)
top_10_table_path = os.path.join(output_dir, 'top_10_nodes_by_degree.csv')
top_10_df.to_csv(top_10_table_path, index=False)
print(f"Top 10 nodes by degree centrality saved to {top_10_table_path}")


# Group nodes by solo converting list of edges into a list of adjacency
solo_adjacency = {}

# Iterate through edges in the graph
for source, target, data in graph.edges(data=True):
    weight = data.get('weight', 1)  

    # Add weight to solo-level adjacency
    if source not in solo_adjacency:
        solo_adjacency[source] = {}
    if target not in solo_adjacency[source]:
        solo_adjacency[source][target] = 0
    solo_adjacency[source][target] += weight

# Save solo adjacency as DataFrame
solo_adjacency_data = [
    {'Source': source, 'Target': target, 'Weight': total_weight}
    for source, targets in solo_adjacency.items()
    for target, total_weight in targets.items()
]
solo_adjacency_df = pd.DataFrame(solo_adjacency_data)

# Save solo adjacency list
solo_adjacency_path = os.path.join(output_dir, 'solo_adjacency_list.csv')
solo_adjacency_df.to_csv(solo_adjacency_path, index=False)
print(f"Solo adjacency list saved to {solo_adjacency_path}")


# Group nodes by performer
performer_influences = {} #initiate empty to store weights

# Map solos to performers
node_to_performer = {node: melid_to_labels[node]['performer'] for node in graph.nodes if node in melid_to_labels}

for _, row in solo_adjacency_df.iterrows():
    source_solo, target_solo, weight = row['Source'], row['Target'], row['Weight']

    # Map solos to performers
    source_performer = node_to_performer.get(source_solo, None)
    target_performer = node_to_performer.get(target_solo, None)

    # Skip if performer is not found
    if not source_performer or not target_performer:
        continue

    # Add weights to influences
    if source_performer not in performer_influences:
        performer_influences[source_performer] = {}
    if target_performer not in performer_influences[source_performer]:
        performer_influences[source_performer][target_performer] = 0
    performer_influences[source_performer][target_performer] += weight

# Convert performer adjacency to a DataFrames
performer_adjacency_data = [
    {'Source': source, 'Target': target, 'Weight': total_weight}
    for source, targets in performer_influences.items()
    for target, total_weight in targets.items()
    if source != target  # Exclude self-loops 
]
performer_adjacency_df = pd.DataFrame(performer_adjacency_data)

# Save performer adjacency list
performer_adjacency_path = os.path.join(output_dir, 'performer_adjacency_list.csv')
performer_adjacency_df.to_csv(performer_adjacency_path, index=False)
print(f"Performer adjacency list saved to {performer_adjacency_path}")

# Visualise performer network
performer_graph = nx.DiGraph()

for _, row in performer_adjacency_df.iterrows():
    performer_graph.add_edge(row['Source'], row['Target'], weight=row['Weight'])

plt.figure(figsize=(16, 16))
layout = nx.spring_layout(performer_graph, seed=42, k=2.2)
node_sizes = [3000 * nx.degree_centrality(performer_graph)[node] for node in performer_graph.nodes()]
nx.draw(
    performer_graph,
    pos=layout,
    with_labels=False,
    node_size=node_sizes,
    node_color='#1f78b4',
    edge_color='gray',
    alpha=0.9,
    font_size=10,
    font_color='black',
    linewidths=7.5,
    width=0.75
)

# Add conditions to facilitate reading of labels
node_labels = {node: node for node in performer_graph.nodes()}  # Use performer names as labels
nx.draw_networkx_labels(
    performer_graph,
    pos=layout,
    labels=node_labels,
    font_size=9,
    font_color='black',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2',alpha=0.75)
)
plt.title("Performer Influence Network", fontsize=14)
plt.tight_layout()
plt.show()

# Get degree value for each performer and save
performer_degree = dict(performer_graph.degree(weight=None))  # Degree without considering edge weights
performer_degree_df = pd.DataFrame(
    performer_degree.items(),
    columns=['Performer', 'Connections']
)
degree_path = os.path.join(output_dir, 'performer_degree.csv')
performer_degree_df.to_csv(degree_path, index=False)
print(f"Performer degree data saved to {degree_path}")

# Plot bar chart of performers by degree
performer_degree_sorted_df = performer_degree_df.sort_values(by='Connections', ascending=False)
plt.figure(figsize=(12, 8))
plt.barh(
    performer_degree_sorted_df['Performer'],
    performer_degree_sorted_df['Connections'],
    color='#1f78b4',
    edgecolor='black',
    alpha=0.8
)
plt.title("Number of Connections by Performer", fontsize=14)
plt.xlabel("Number of Connections")
plt.ylabel("Performer")
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.show()

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/network/network_solos_analysis.py
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import argparse

# Parse command-line arguments
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/network/network_patterns_analysis.py
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

# Add descriptions for filtered parameters
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/network/network_patterns.py
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, '../../data/output/mcmc_results/')
best_network_params_file = 'best_network_patterns_params.csv'
metadata_path = os.path.join(base_dir, '../../data/solos_db_with_durations_cleaned.csv')
sequence_patterns_path = os.path.join(base_dir, '../../data/sequences_patterns.csv')
top_patterns_sample_path = os.path.join(base_dir, '../../data/top_patterns_sample.csv')

# Load required data
metadata = pd.read_csv(metadata_path)
sequence_patterns = pd.read_csv(sequence_patterns_path)
metadata_cleaned = metadata[['melid', 'title', 'performer']].drop_duplicates(subset=['melid'])

# Map pattern_id to melid and metadata
pattern_to_melid = sequence_patterns.set_index('pattern_id')['melid'].to_dict()
melid_to_title = metadata_cleaned.set_index('melid')['title'].to_dict()
melid_to_performer = metadata_cleaned.set_index('melid')['performer'].to_dict()

# Load network edges
best_network_params_path = os.path.join(output_dir, best_network_params_file)
best_params_df = pd.read_csv(best_network_params_path)
if best_params_df.empty:
    raise ValueError("The best network parameters file is empty.")
best_parameters = best_params_df.iloc[0, 0]
final_network_file = f"final_network_patterns_{best_parameters}.csv"
final_network_path = os.path.join(output_dir, final_network_file)
edges_df = pd.read_csv(final_network_path)

# Enrich edges_df with metadata
edges_df['source_melid'] = edges_df['node_a'].map(pattern_to_melid)
edges_df['source_title'] = edges_df['source_melid'].map(melid_to_title)
edges_df['source_performer'] = edges_df['source_melid'].map(melid_to_performer)
edges_df['target_melid'] = edges_df['node_b'].map(pattern_to_melid)
edges_df['target_title'] = edges_df['target_melid'].map(melid_to_title)
edges_df['target_performer'] = edges_df['target_melid'].map(melid_to_performer)

# Drop self-loops
edges_df = edges_df[edges_df['source_melid'] != edges_df['target_melid']]

# Check for missing mappings
unmapped_sources = edges_df['source_performer'].isnull().sum()
unmapped_targets = edges_df['target_performer'].isnull().sum()
print(f"Unmapped Sources: {unmapped_sources}, Unmapped Targets: {unmapped_targets}")

# Build adjacency list by performer
performer_influences = {}
for _, row in edges_df.iterrows():
    source_performer = row['source_performer']
    target_performer = row['target_performer']
    weight = row['weight']

    if pd.isnull(source_performer) or pd.isnull(target_performer):
        continue  # Skip rows with missing performer mappings
    # Skip self-loops
    if source_performer == target_performer:
        continue

    if source_performer not in performer_influences:
        performer_influences[source_performer] = {}
    if target_performer not in performer_influences[source_performer]:
        performer_influences[source_performer][target_performer] = 0
    performer_influences[source_performer][target_performer] += weight

# Convert performer influences to DataFrame
performer_adjacency_data = [
    {'Source': source, 'Target': target, 'Weight': weight}
    for source, targets in performer_influences.items()
    for target, weight in targets.items()
]
performer_adjacency_df = pd.DataFrame(performer_adjacency_data)

# Save performer adjacency list
performer_adjacency_path = os.path.join(output_dir, 'performer_adjacency_list.csv')
performer_adjacency_df.to_csv(performer_adjacency_path, index=False)
print(f"Performer adjacency list saved to {performer_adjacency_path}")

# Visualise performer network
if not performer_adjacency_df.empty:
    performer_graph = nx.DiGraph()
    for _, row in performer_adjacency_df.iterrows():
        performer_graph.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    plt.figure(figsize=(16, 16))
    layout = nx.spring_layout(performer_graph, seed=42)
    node_sizes = [3000 * nx.degree_centrality(performer_graph).get(node, 0) for node in performer_graph.nodes()]
    nx.draw(
    performer_graph,
    pos=layout,
    with_labels=False,
    node_size=node_sizes,
    node_color='#1f78b4',
    edge_color='gray',
    alpha=0.9,
    font_size=10,
    font_color='black',
    linewidths=7.5,
    width=0.75
    )
    # Add conditions to facilitate reading of labels
    node_labels = {node: node for node in performer_graph.nodes()}  # Use performer names as labels
    nx.draw_networkx_labels(
    performer_graph,
    pos=layout,
    labels=node_labels,
    font_size=9,
    font_color='black',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2',alpha=0.75)
)
    plt.title("Performer Influence Network", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("No data to plot for Performer Influence Network.")

# Save degree data
performer_degree = dict(performer_graph.degree(weight=None))
performer_degree_df = pd.DataFrame(
    performer_degree.items(),
    columns=['Performer', 'Connections']
)
degree_path = os.path.join(output_dir, 'performer_degree.csv')
performer_degree_df.to_csv(degree_path, index=False)
print(f"Performer degree data saved to {degree_path}")
# Sort performers by connections
performer_degree_sorted_df = performer_degree_df.sort_values(by='Connections', ascending=False)
print("Performer Adjacency DataFrame:")
print(performer_adjacency_df.head())
print(f"Number of rows in performer_adjacency_df: {len(performer_adjacency_df)}")

print("Performer Degree DataFrame:")
print(performer_degree_df.head())
print(f"Number of rows in performer_degree_df: {len(performer_degree_df)}")
# Plot bar chart
plt.figure(figsize=(12, 8))
plt.barh(
    performer_degree_sorted_df['Performer'],
    performer_degree_sorted_df['Connections'],
    color='#1f78b4',
    edgecolor='black',
    alpha=0.8
)
plt.title("Number of Connections by Performer", fontsize=14)
plt.xlabel("Number of Connections")
plt.ylabel("Performer")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/gen_ini_net_solos_dir.py
import numpy as np
import pandas as pd
import random
import networkx as nx
import os

# Define relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))
prior_matrix_path = os.path.join(base_dir, '../../data/output/prior_matrix_solos.npy')
output_csv_path = os.path.join(base_dir, '../../data/output/initial_network_solos.csv')
metrics_node_csv_path = os.path.join(base_dir, '../../data/output/initial_network_metrics_nodes.csv')
metrics_global_csv_path = os.path.join(base_dir, '../../data/output/initial_network_metrics_global.csv')

# Load the prior matrix
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

solo_id_list = metadata_cleaned['melid'].tolist()


# Validate the alignment with the prior matrix
if prior_matrix_solos.shape[0] != len(solo_id_list):
    raise ValueError(
        f"Mismatch: Prior matrix size {prior_matrix_solos.shape[0]} "
        f"does not match the number of unique solo IDs {len(solo_id_list)}."
    )

melid_to_index = {melid: idx for idx, melid in enumerate(solo_id_list)}
# Replace solo_id_list usage with the corresponding `melid` values
# Function to generate an initial network with weights

if prior_matrix_solos.shape[0] != len(solo_id_list):
    raise ValueError(
        f"Mismatch: Prior matrix size {prior_matrix_solos.shape[0]} "
        f"does not match the number of solo IDs {len(solo_id_list)}."
    )
def generate_initial_network_with_weights(solo_id_list, prior_matrix, random_selection=True, seed_value=42):
    """
    Generates an initial directed network using a prior matrix.
    
    Parameters:
        solo_id_list (list): List of solo IDs corresponding to the prior matrix dimensions.
        prior_matrix (np.ndarray): Prior matrix used to define edge weights.
        random_selection (bool): If True, selects a random predecessor. Otherwise, uses the highest value.
        seed_value (int): Seed for reproducibility of random operations.

    Returns:
        nx.DiGraph: Generated directed network.
    """
    # Set the random seed for reproducibility
    random.seed(seed_value)

    # Create an empty directed graph and add all nodes
    network = nx.DiGraph()
    network.add_nodes_from(solo_id_list)
    melid_to_index = {melid: idx for idx, melid in enumerate(solo_id_list)}
    index_to_melid = {idx: melid for melid, idx in melid_to_index.items()}
    matrix_size = prior_matrix.shape[0]

    for i in range(matrix_size):
        valid_predecessors = [j for j in range(matrix_size) if prior_matrix[j, i] == 1]
        print(f"Node {index_to_melid[i]}: Valid predecessors: {[index_to_melid[j] for j in valid_predecessors]}")
        
        if not valid_predecessors:
            continue
        
        if random_selection:
            predecessor = random.choice(valid_predecessors)
        else:
            predecessor = max(valid_predecessors, key=lambda x: prior_matrix[x, i])
        
        if predecessor == i:
            continue
        
        network.add_edge(index_to_melid[predecessor], index_to_melid[i], weight=1)
        print(f"Adding edge from {index_to_melid[predecessor]} to {index_to_melid[i]} with weight 1")
    
    return network

# Generate the initial network
initial_network = generate_initial_network_with_weights(solo_id_list, prior_matrix_solos)

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
print(f"Initial network saved to {output_csv_path}")

# --- Compute Metrics ---

# Node-level metrics
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/mcmc_sampling_patterns.py
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/gen_ini_net_patterns_u.py
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/gen_ini_net_solos_u.py
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/priors_matrix_patterns.py
import numpy as np
import pandas as pd

# Load similarity matrix to get aligned pattern IDs
similarity_matrix_df = pd.read_csv('data/output/similarity_matrix_patterns_k2_0.75.csv', index_col=0)
aligned_pattern_ids = similarity_matrix_df.index.astype(int).unique()  # Use the unnamed column as index
print(f"Aligned Pattern IDs: {aligned_pattern_ids}")

# Load top_patterns_sample to get valid pattern IDs and corresponding melids
top_patterns_sample_df = pd.read_csv('data/top_patterns_sample.csv')

# Filter top_patterns_sample to only include the aligned pattern IDs
filtered_top_patterns = top_patterns_sample_df[top_patterns_sample_df['pattern_id'].isin(aligned_pattern_ids)]
print(f"Filtered Top Patterns Shape: {filtered_top_patterns.shape}")

# Extract the required columns: pattern_id and melid
aligned_patterns_with_melids = filtered_top_patterns[['pattern_id', 'melid']].drop_duplicates()

# Load solos dataset to get metadata for the aligned melid values
solos_df = pd.read_csv('data/solos_db_with_durations_cleaned.csv')

# Filter solos dataset to only include the aligned melid values
filtered_solos = solos_df[solos_df['melid'].isin(aligned_patterns_with_melids['melid'])]
print(f"Filtered Solos Shape: {filtered_solos.shape}")

# Extract relevant columns: melid, performer, title, release_date
filtered_solos = filtered_solos[['melid', 'performer', 'title', 'releasedate']]

# Sort the solos based on release date to maintain chronological order
filtered_solos.sort_values(by='releasedate', inplace=True)

# Create a dictionary to map melid to release date for easy lookup
melid_to_release_date = filtered_solos.set_index('melid')['releasedate'].to_dict()

# Create a mapping from pattern_id to melid using aligned_patterns_with_melids
pattern_id_to_melid = aligned_patterns_with_melids.set_index('pattern_id')['melid'].to_dict()

# Create an empty prior matrix based on the number of aligned patterns
valid_pattern_ids = aligned_patterns_with_melids['pattern_id'].unique()
num_aligned_patterns = len(valid_pattern_ids)
prior_matrix = np.zeros((num_aligned_patterns, num_aligned_patterns))

# Create a mapping from pattern_id to index in the prior matrix
pattern_id_to_index = {pattern_id: i for i, pattern_id in enumerate(valid_pattern_ids)}

# Iterate through each pair of pattern_ids to determine temporal influence based on melid release dates
for pattern_id_a in valid_pattern_ids:
    melid_a = pattern_id_to_melid[pattern_id_a]
    release_date_a = melid_to_release_date[melid_a]

    for pattern_id_b in valid_pattern_ids:
        melid_b = pattern_id_to_melid[pattern_id_b]
        release_date_b = melid_to_release_date[melid_b]

        # If melid_a has an earlier release date than melid_b, it can be an influence
        if release_date_a < release_date_b:
            prior_matrix[pattern_id_to_index[pattern_id_a], pattern_id_to_index[pattern_id_b]] = 1

print(f"Prior Matrix Shape: {prior_matrix.shape}")

# Save the prior matrix to a file
prior_matrix_path = 'data/output/prior_matrix_patterns.npy'
np.save(prior_matrix_path, prior_matrix)

print(f"Prior matrix saved to {prior_matrix_path}")


# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/ini_net_solos.py
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Network Analysis")
parser.add_argument("--base_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Base directory for relative paths")
args = parser.parse_args()
base_dir = args.base_dir

# Define relative paths
node_metrics_files = {
    'Directed': os.path.join(base_dir, '../../data/output/initial_network_metrics_nodes.csv'),
    'Undirected': os.path.join(base_dir, '../../data/output/initial_network_metrics_nodes_undirected.csv'),
    'Multigraph': os.path.join(base_dir, '../../data/output/initial_multigraph_metrics_nodes.csv')
}
# Define paths to the graph files
directed_graph_path = os.path.join(base_dir, '../../data/output/initial_network_solos.csv')
undirected_graph_path = os.path.join(base_dir, '../../data/output/initial_network_solos_undirected.csv')
multigraph_path = os.path.join(base_dir, '../../data/output/initial_multigraph_solos.csv')

# Load graphs
def load_graph(path, graph_type):
    edges_df = pd.read_csv(path, delimiter=',', header=0)
    if graph_type == "Multigraph":
        edges_df['weight'] = pd.to_numeric(edges_df['weight'], errors='coerce')
        edges_df = edges_df.dropna(subset=['weight'])
        return nx.from_pandas_edgelist(edges_df, source='source', target='target', edge_attr='weight', create_using=nx.MultiGraph())
    elif graph_type == "Undirected":
        return nx.from_pandas_edgelist(edges_df, source='source', target='target', edge_attr='weight', create_using=nx.Graph())
    else:
        return nx.from_pandas_edgelist(edges_df, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

initial_directed = load_graph(directed_graph_path, "Directed")
initial_undirected = load_graph(undirected_graph_path, "Undirected")
initial_multigraph = load_graph(multigraph_path, "Multigraph")

# Load node metrics
node_metrics = []
for graph_type, file_path in node_metrics_files.items():
    df = pd.read_csv(file_path)
    df['graph_type'] = graph_type
    node_metrics.append(df)

node_metrics_df = pd.concat(node_metrics, ignore_index=True)
output_node_metrics_file = os.path.join(base_dir, '../../data/output/combined_node_metrics.csv')
node_metrics_df.to_csv(output_node_metrics_file, index=False)
print(f"Combined node metrics saved to {output_node_metrics_file}")

# Plot degree centrality
plt.figure(figsize=(12, 6))
sns.boxplot(x='graph_type', y='degree_centrality', data=node_metrics_df, palette='Set2', hue=None)
plt.title('Degree Centrality Across Graph Types')
plt.xlabel('Graph Type')
plt.ylabel('Degree Centrality')
plt.tight_layout()
plt.show()

# Plot closeness centrality
plt.figure(figsize=(12, 6))
sns.boxplot(x='graph_type', y='closeness_centrality', data=node_metrics_df, palette='Set2', hue=None)
plt.title('Closeness Centrality Across Graph Types')
plt.xlabel('Graph Type')
plt.ylabel('Closeness Centrality')
plt.tight_layout()
plt.show()

# Plot betweenness centrality
plt.figure(figsize=(12, 6))
sns.boxplot(x='graph_type', y='betweenness_centrality', data=node_metrics_df, palette='Set2', hue=None)
plt.title('Betweenness Centrality Across Graph Types')
plt.xlabel('Graph Type')
plt.ylabel('Betweenness Centrality')
plt.tight_layout()
plt.show()

# Highlight top nodes in visualization
top_nodes_summary = {}
for graph_type in node_metrics_files.keys():
    top_nodes = (
        node_metrics_df[node_metrics_df['graph_type'] == graph_type]
        .nlargest(5, 'degree_centrality')[['node', 'degree_centrality']]
    )
    top_nodes_summary[graph_type] = top_nodes

# Combine top nodes for all graph types into a single DataFrame
top_nodes_df = pd.concat(
    [
        df.assign(
            graph_type=graph_type,
            degree=[
                initial_multigraph.degree(node) if graph_type == 'Multigraph'
                else initial_undirected.degree(node) if graph_type == 'Undirected'
                else initial_directed.in_degree(node) + initial_directed.out_degree(node)
                for node in df['node']
            ]
        )
        for graph_type, df in top_nodes_summary.items()
    ],
    ignore_index=True
)
top_nodes_output_file = os.path.join(base_dir, '../../data/output/top_nodes_by_degree_centrality.csv')
top_nodes_df.to_csv(top_nodes_output_file, index=False)
print(f"Top nodes by degree centrality saved to {top_nodes_output_file}")

# Load additional data for labels
solos_db_path = os.path.join(base_dir, '../../data/solos_db_with_durations_cleaned.csv')
solos_db_df = pd.read_csv(solos_db_path)

# Subset the metadata to only include melid, title, and performer
solos_db_subset = solos_db_df[['melid', 'title', 'performer']]

# Drop duplicates based on melid
solos_db_subset = solos_db_subset.drop_duplicates(subset=['melid'])

# Ensure no duplicates exist for melid
if solos_db_subset['melid'].duplicated().any():
    raise ValueError("Duplicate melid entries found after cleaning!")

# Print the cleaned metadata
print(solos_db_subset.head())

# Create the melid_to_labels dictionary
melid_to_labels = solos_db_subset.set_index('melid')[['title', 'performer']].to_dict('index')

# Convert melid column to a list only if needed later
melid_list = solos_db_subset['melid'].tolist()


melid_to_labels = solos_db_subset.set_index('melid')[['title', 'performer']].to_dict('index')

graph_nodes = set(initial_directed.nodes) | set(initial_undirected.nodes) | set(initial_multigraph.nodes)
db_melids = set(solos_db_df['melid'])

# Define consistent layout for graphs
layout = nx.spring_layout(initial_directed, seed=42,k=0.9)

# Visualize graphs
# Visualize graphs with unified colours and labels only for top nodes
from adjustText import adjust_text

# Visualize graphs with unified colours and labels only for top nodes
fig, axes = plt.subplots(1, 3, figsize=(20, 9))

for ax, graph, graph_type, title in zip(
    axes,
    [initial_directed, initial_undirected, initial_multigraph],
    ['Directed', 'Undirected', 'Multigraph'],
    ['Directed Graph', 'Undirected Graph', 'Multigraph']
):
    top_nodes = top_nodes_summary[graph_type]['node'].values  # Get top nodes for the current graph type
    
    # Create labels for the nodes
    node_labels = {
        node: f"{melid_to_labels[node]['title']} ({melid_to_labels[node]['performer']})"
        if node in top_nodes else ''
        for node in graph.nodes
    }

    # Adjust node sizes and colours
    node_colours = ['#f97c7c' if node in top_nodes else '#5588ff' for node in graph.nodes]
    node_sizes = [1000 * nx.degree_centrality(graph)[node] for node in graph.nodes]

    # Draw the graph
    nx.draw(
        graph,
        pos=layout,
        ax=ax,
        with_labels=False,  # Disable default labels
        node_color=node_colours,
        edge_color='gray',
        alpha=0.7,  # Consistent transparency
        node_size=node_sizes,
    )
    
    # Add labels manually to avoid overlaps
    texts = []
    for node in graph.nodes:
        if node in top_nodes:
            x, y = layout[node]
            label = node_labels[node]
            texts.append(ax.text(x, y, label, fontsize=10, ha='center', va='center', fontweight='bold'))

    # Adjust text positions to avoid overlap
    adjust_text(texts, ax=ax)

    ax.set_title(f"{title} (Top Nodes Highlighted)")

plt.tight_layout()
plt.show()

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/gen_ini_net_patterns_dir.py
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/gen_ini_net_patterns_multi.py
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

pattern_id_list = filtered_pattern_ids  # Use the filtered list of pattern ids

# Generate initial network
def generate_initial_multigraph_with_weights(pattern_id_list, prior_matrix):
    multigraph = nx.MultiDiGraph()
    multigraph.add_nodes_from(pattern_id_list)

    matrix_size = prior_matrix.shape[0]  

    for i in range(matrix_size):
        valid_predecessors = [
            j for j in range(matrix_size)
            if prior_matrix[j, i] > 0
        ]
        frequency_weight = len(valid_predecessors) / matrix_size  # Normalised weight

        for predecessor in valid_predecessors:
            weight = max(1e-5, frequency_weight)  # Control to keep weights as positive
            print(f"Adding edge from {pattern_id_list[predecessor]} to {pattern_id_list[i]} with weight {weight}")
            multigraph.add_edge(pattern_id_list[predecessor], pattern_id_list[i], weight=weight)

    return multigraph

# Generate initial network
initial_multigraph = generate_initial_multigraph_with_weights(pattern_id_list, filtered_prior_matrix)

# Save the initial network to a csv file
output_path = os.path.join(base_dir, '../../data/output/initial_multigraph_patterns.csv')
edges_data = [
    {'source': edge[0], 'target': edge[1], 'weight': edge[2]['weight']}
    for edge in initial_multigraph.edges(data=True)
]
edges_df = pd.DataFrame(edges_data)
edges_df.to_csv(output_path, index=False)
print(f"Initial multigraph saved to {output_path}")

# Calculate Metrics

# Node metrics
metrics = {
    'node': list(initial_multigraph.nodes),
    'in_degree': [val for _, val in initial_multigraph.in_degree()],
    'out_degree': [val for _, val in initial_multigraph.out_degree()],
    'betweenness_centrality': list(nx.betweenness_centrality(initial_multigraph).values()),
    'closeness_centrality': list(nx.closeness_centrality(initial_multigraph).values()),
    'pagerank': list(nx.pagerank(initial_multigraph).values()),
    'degree_centrality': list(nx.degree_centrality(initial_multigraph).values())
}

metrics_df = pd.DataFrame(metrics)

#  Output path for metrics
metrics_node_csv_path = os.path.join(base_dir, '../../data/output/node_metrics_patterns_multigraph.csv')
metrics_df.to_csv(metrics_node_csv_path, index=False)
print(f"Node-level metrics saved to {metrics_node_csv_path}")

# Global metrics
global_metrics = {
    'number_of_nodes': [initial_multigraph.number_of_nodes()],
    'number_of_edges': [initial_multigraph.number_of_edges()],
    'average_degree': [sum(dict(initial_multigraph.degree()).values()) / initial_multigraph.number_of_nodes()],
    'density': [nx.density(initial_multigraph)]
}

global_metrics_df = pd.DataFrame(global_metrics)

# Output path for metrics
metrics_global_csv_path = os.path.join(base_dir, '../../data/output/global_metrics_patterns_multigraph.csv')
global_metrics_df.to_csv(metrics_global_csv_path, index=False)
print(f"Global-level metrics saved to {metrics_global_csv_path}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/generate_initial_network.py
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/gen_ini_net_solos_multi.py
import numpy as np
import pandas as pd
import random
import networkx as nx
import os

# Define relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))
prior_matrix_path = os.path.join(base_dir, '../../data/output/prior_matrix_solos.npy')
output_csv_path = os.path.join(base_dir, '../../data/output/initial_multigraph_solos.csv')
metrics_node_csv_path = os.path.join(base_dir, '../../data/output/initial_multigraph_metrics_nodes.csv')
metrics_global_csv_path = os.path.join(base_dir, '../../data/output/initial_multigraph_metrics_global.csv')

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

# Function to generate an initial multigraph with weights
def generate_initial_multigraph_with_weights(solo_id_list, prior_matrix, random_selection=True, seed_value=42, max_edges=3):
    """
    Generates an initial directed multigraph using a prior matrix.
    
    Parameters:
        solo_id_list (list): List of solo IDs corresponding to the prior matrix dimensions.
        prior_matrix (np.ndarray): Prior matrix used to define edge weights.
        random_selection (bool): If True, selects a random predecessor. Otherwise, uses the highest value.
        seed_value (int): Seed for reproducibility of random operations.
        max_edges (int): Maximum number of edges allowed between two nodes.

    Returns:
        nx.MultiDiGraph: Generated directed multigraph.
    """
    # Set the random seed for reproducibility
    random.seed(seed_value)

    # Create an empty directed multigraph and add all nodes
    multigraph = nx.MultiDiGraph()
    multigraph.add_nodes_from(solo_id_list)

    matrix_size = prior_matrix.shape[0]

    for i in range(matrix_size):
        # Find valid predecessors (those with a value of 1 in the prior matrix)
        valid_predecessors = [j for j in range(matrix_size) if prior_matrix[j, i] == 1]

        # If there are no valid predecessors, skip the current node
        if not valid_predecessors:
            continue

        # Add multiple edges
        num_edges = random.randint(1, max_edges)  # Randomly decide how many edges to add
        for _ in range(num_edges):
            if random_selection:
                predecessor = random.choice(valid_predecessors)
            else:
                predecessor = max(valid_predecessors, key=lambda x: prior_matrix[x, i])

            # Add an edge with a random weight
            weight = random.uniform(0.5, 2.0)  # Example random weight range
            multigraph.add_edge(solo_id_list[predecessor], solo_id_list[i], weight=weight)

            print(f"Adding edge from {solo_id_list[predecessor]} to {solo_id_list[i]} with weight {weight:.2f}")

    return multigraph

# Generate the initial multigraph
initial_multigraph = generate_initial_multigraph_with_weights(solo_id_list, prior_matrix_solos)

# Add final validation step before saving
for node in initial_multigraph.nodes():
    if node not in solo_id_list:
        raise ValueError(f"Node {node} is not a valid solo ID. Check the prior matrix dimensions.")

# Convert the multigraph edges to a DataFrame
edges_data = [
    {'source': edge[0], 'target': edge[1], 'weight': edge[2]['weight']}
    for edge in initial_multigraph.edges(data=True)
]
edges_df = pd.DataFrame(edges_data)

# Save the edges as a CSV file
edges_df.to_csv(output_csv_path, index=False)
print(f"Initial multigraph saved to {output_csv_path}")

# --- Compute Metrics ---

# Node-level metrics
metrics = {
    'node': list(initial_multigraph.nodes),
    'in_degree': [val for _, val in initial_multigraph.in_degree()],
    'out_degree': [val for _, val in initial_multigraph.out_degree()],
    'betweenness_centrality': list(nx.betweenness_centrality(initial_multigraph).values()),
    'closeness_centrality': list(nx.closeness_centrality(initial_multigraph).values()),
    'pagerank': list(nx.pagerank(initial_multigraph).values()),
    'degree_centrality': list(nx.degree_centrality(initial_multigraph).values())  # Add this line
}

metrics_df = pd.DataFrame(metrics)

# Save node-level metrics
metrics_df.to_csv(metrics_node_csv_path, index=False)
print(f"Node-level metrics saved to {metrics_node_csv_path}")

# Global-level metrics
global_metrics = {
    'number_of_nodes': [initial_multigraph.number_of_nodes()],
    'number_of_edges': [initial_multigraph.number_of_edges()],
    'average_degree': [sum(dict(initial_multigraph.degree()).values()) / initial_multigraph.number_of_nodes()],
    'density': [nx.density(initial_multigraph)]
}

global_metrics_df = pd.DataFrame(global_metrics)

# Save global-level metrics
global_metrics_df.to_csv(metrics_global_csv_path, index=False)
print(f"Global-level metrics saved to {metrics_global_csv_path}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/generate_prior_matrices.py
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/agg_net_pattern_d.py
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
aggregated_path = os.path.join(base_dir, '../../data/output/aggregated_patterns_directed.csv')
metadata_path = os.path.join(base_dir, '../../data/solos_db_with_durations_cleaned.csv')
sequence_patterns_path = os.path.join(base_dir, '../../data/sequences_patterns.csv')

aggregated_data = pd.read_csv(aggregated_path)
metadata = pd.read_csv(metadata_path)
sequence_patterns = pd.read_csv(sequence_patterns_path)
metadata_cleaned = metadata[['melid', 'title', 'performer']].drop_duplicates(subset=['melid'])

# Map pattern_id to melid to get metadata
pattern_to_melid = sequence_patterns.set_index('pattern_id')['melid'].to_dict()
melid_to_title = metadata_cleaned.set_index('melid')['title'].to_dict()

# Map pattern_id to title
aggregated_data['source_melid'] = aggregated_data['source'].map(pattern_to_melid)
aggregated_data['source_title'] = aggregated_data['source_melid'].map(melid_to_title)
aggregated_data['target_melids'] = aggregated_data['targets'].apply(
    lambda targets: [pattern_to_melid.get(t) for t in eval(targets)]
)
aggregated_data['target_titles'] = aggregated_data['target_melids'].apply(
    lambda melids: [melid_to_title.get(m) for m in melids]
)
# Map performer to titles
melid_to_performer = metadata_cleaned.set_index('melid')['performer'].to_dict()
aggregated_data['source_performer'] = aggregated_data['source_melid'].map(melid_to_performer)
aggregated_data['target_performers'] = aggregated_data['target_melids'].apply(
    lambda melids: [melid_to_performer.get(m) for m in melids]
)
print(aggregated_data[['source_title', 'target_titles', 'total_weight']].head())


# Generate directed graph
directed_graph = nx.DiGraph()

for _, row in aggregated_data.explode('target_titles').iterrows():
    directed_graph.add_edge(row['source_title'], row['target_titles'], weight=row['total_weight'])

# Save the directed graph
directed_output_path = os.path.join(base_dir, '../../data/output/aggregated_patterns_directed_graph.csv')
nx.write_weighted_edgelist(directed_graph, directed_output_path)
print(f"Directed graph saved to {directed_output_path}")

# Generate undirected graph
undirected_graph = nx.Graph()

for _, row in aggregated_data.explode('target_titles').iterrows():
    source = row['source_title']
    target = row['target_titles']
    weight = row['total_weight']

    if undirected_graph.has_edge(source, target):
        undirected_graph[source][target]['weight'] += weight
    else:
        undirected_graph.add_edge(source, target, weight=weight)

# Save undirected graph
undirected_output_path = os.path.join(base_dir, '../../data/output/aggregated_patterns_undirected.csv')
nx.write_weighted_edgelist(undirected_graph, undirected_output_path)
print(f"Undirected graph saved to {undirected_output_path}")

# Generate Multigraph
multigraph = nx.MultiDiGraph()

for _, row in aggregated_data.iterrows():
    source = row['source_title']
    targets = row['target_titles']
    total_weight = row['total_weight']

    for target in targets:
        multigraph.add_edge(source, target, weight=total_weight / len(targets))

# Save the multigraph
multigraph_output_path = os.path.join(base_dir, '../../data/output/aggregated_patterns_multigraph.csv')
edges = [
    {'source': u, 'target': v, 'weight': data['weight']}
    for u, v, data in multigraph.edges(data=True)
]
pd.DataFrame(edges).to_csv(multigraph_output_path, index=False)
print(f"Multigraph saved to {multigraph_output_path}")

# Metrics
def compute_metrics(graph, metrics_path, global_path, graph_name="Graph"):
    if isinstance(graph, nx.MultiDiGraph) or isinstance(graph, nx.MultiGraph):
        node_metrics = {
            'node': list(graph.nodes),
            'in_degree': [val for _, val in graph.in_degree()] if isinstance(graph, nx.MultiDiGraph) else None,
            'out_degree': [val for _, val in graph.out_degree()] if isinstance(graph, nx.MultiDiGraph) else None,
            'degree': [val for _, val in graph.degree()],
            'degree_centrality': list(nx.degree_centrality(graph).values())
        }
        # Exception for multigraphs cannot compute clustering metrics or connected components 
        connected_components = 'N/A'
        average_clustering = 'N/A'
    elif isinstance(graph, nx.DiGraph):
        node_metrics = {
            'node': list(graph.nodes),
            'in_degree': [val for _, val in graph.in_degree()],
            'out_degree': [val for _, val in graph.out_degree()],
            'degree_centrality': list(nx.degree_centrality(graph).values()),
            'closeness_centrality': list(nx.closeness_centrality(graph).values())
        }
        connected_components = nx.number_strongly_connected_components(graph)
        average_clustering = 'N/A'  # Not available for directed graphs
    elif isinstance(graph, nx.Graph):
        node_metrics = {
            'node': list(graph.nodes),
            'degree': [val for _, val in graph.degree()],
            'degree_centrality': list(nx.degree_centrality(graph).values()),
            'closeness_centrality': list(nx.closeness_centrality(graph).values())
        }
        connected_components = nx.number_connected_components(graph)
        average_clustering = nx.average_clustering(graph)
    else:
        raise ValueError(f"Unsupported graph type: {type(graph)}")

    metrics_df = pd.DataFrame(node_metrics)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"{graph_name} node metrics saved to {metrics_path}")

    global_metrics = {
        'number_of_nodes': [graph.number_of_nodes()],
        'number_of_edges': [graph.number_of_edges()],
        'density': [nx.density(graph)],
        'average_degree_centrality': [sum(nx.degree_centrality(graph).values()) / graph.number_of_nodes()],
        'average_clustering': [average_clustering],
        'connected_components': [connected_components]
    }
    global_metrics_df = pd.DataFrame(global_metrics)
    global_metrics_df.to_csv(global_path, index=False)
    print(f"{graph_name} global metrics saved to {global_path}")

    return metrics_df 

# Metrics for Directed Graph
directed_metrics_df = compute_metrics(
    directed_graph,
    os.path.join(base_dir, '../../data/output/node_metrics_directed_aggregated.csv'),
    os.path.join(base_dir, '../../data/output/global_metrics_directed_aggregated.csv'),
    "Directed Graph"
)

# Metrics for Undirected Graph
undirected_metrics_df = compute_metrics(
    undirected_graph,
    os.path.join(base_dir, '../../data/output/node_metrics_undirected_aggregated.csv'),
    os.path.join(base_dir, '../../data/output/global_metrics_undirected_aggregated.csv'),
    "Undirected Graph"
)

# Metrics for Multigraph
multigraph_metrics_df = compute_metrics(
    multigraph,
    os.path.join(base_dir, '../../data/output/node_metrics_multigraph_aggregated.csv'),
    os.path.join(base_dir, '../../data/output/global_metrics_multigraph_aggregated.csv'),
    "Multigraph"
)

# Calculate total degree for directed graph
directed_metrics_df['total_degree'] = (
    directed_metrics_df['in_degree'] + directed_metrics_df['out_degree']
)

# Calculate total degree for undirected graph
undirected_metrics_df['total_degree'] = undirected_metrics_df['degree']

# Combine both metrics
combined_metrics = pd.concat([
    directed_metrics_df[['node', 'total_degree']].assign(graph_type='Directed'),
    undirected_metrics_df[['node', 'total_degree']].assign(graph_type='Undirected')
])


# Visualization function for networks
def visualize(graph, metrics_df, metadata, title):
    plt.figure(figsize=(14, 10)) 
    ax = plt.gca()  # Get the current Axes object
    pos = nx.spring_layout(graph, seed=42, k=0.5)

    # # Map performer to titles
    title_to_performer = metadata.set_index('title')['performer'].to_dict()

    # Draw edges with weight colors
    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    edges = nx.draw_networkx_edges(
        graph, pos, edge_color=edge_weights, edge_cmap=plt.cm.Blues, alpha=0.5, ax=ax
    )

    # Add colorbar for edge weights
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Edge Weight')

    # Get top 15 nodes by degree centrality
    top_nodes = metrics_df.nlargest(15, 'degree_centrality')['node']
    node_colors = ['#f97c7c' if node in top_nodes.values else '#5588ff' for node in graph.nodes]
    node_sizes = [500 * nx.degree_centrality(graph).get(node, 0) for node in graph.nodes]

    # Draw nodes and highlight top nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

    # Adjust label positions for top nodes to include performer in the label
    for node, (x, y) in pos.items():
        if node in top_nodes.values:
            performer = title_to_performer.get(node, "Unknown Performer")
            label_x, label_y = x, y + 0.15  # Add offset for clarity of labels
            plt.text(
                label_x,
                label_y,
                f"{node}\n({performer})", 
                fontsize=9,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
                ha='center',
                va='center'
            )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Visualize Directed Graph
visualize(directed_graph, directed_metrics_df, metadata_cleaned, "Directed Graph Visualization")

# Visualize Undirected Graph
visualize(undirected_graph, undirected_metrics_df, metadata_cleaned, "Undirected Graph Visualization")

# Horizontal Histogram for Directed Graph
plt.figure(figsize=(10, 8)) 
top_directed_sorted = directed_metrics_df.nlargest(15, 'total_degree').sort_values(by='total_degree', ascending=True)
plt.barh(top_directed_sorted['node'], top_directed_sorted['total_degree'], color='blue', alpha=0.7)
plt.title('Directed Graph: Top 15 Songs by Total Degree', fontsize=14, fontweight='bold')
plt.xlabel('Total Degree', fontsize=10)
plt.ylabel('Song Title', fontsize=10)
plt.tight_layout()
plt.show()

# Horizontal Histogram for Undirected Graph
plt.figure(figsize=(10, 8)) 
top_undirected_sorted = undirected_metrics_df.nlargest(15, 'total_degree').sort_values(by='total_degree', ascending=True)
plt.barh(top_undirected_sorted['node'], top_undirected_sorted['total_degree'], color='blue', alpha=0.7)
plt.title('Undirected Graph: Top 15 Songs by Total Degree', fontsize=14, fontweight='bold')
plt.xlabel('Total Degree', fontsize=10)
plt.ylabel('Song Title', fontsize=10)
plt.tight_layout()
plt.show()

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/ini_net_patterns.py
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Network Analysis for Patterns")
parser.add_argument("--base_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Base directory for relative paths")
args = parser.parse_args()
base_dir = args.base_dir
# Load sequences_patterns.csv
patterns_path = os.path.join(base_dir, '../../data/sequences_patterns.csv')
patterns_df = pd.read_csv(patterns_path, usecols=['pattern_id', 'melid'])

# Load solos_db_with_durations_cleaned.csv
metadata_path = os.path.join(base_dir, '../../data/solos_db_with_durations_cleaned.csv')
metadata_df = pd.read_csv(metadata_path, usecols=['melid', 'title', 'performer'])

# Merge patterns with metadata
merged_df = patterns_df.merge(metadata_df, on='melid', how='left')

# Check for any missing mappings
missing_mappings = merged_df[merged_df['title'].isnull() | merged_df['performer'].isnull()]
if not missing_mappings.empty:
    print(f"Warning: Some patterns could not be mapped to performer and title. Missing mappings:\n{missing_mappings}")

# Drop duplicates if any exist
merged_df = merged_df.drop_duplicates(subset=['pattern_id'])

# Save the enriched dataset
enriched_patterns_path = os.path.join(base_dir, '../../data/output/enriched_patterns.csv')
merged_df.to_csv(enriched_patterns_path, index=False)
print(f"Enriched patterns saved to {enriched_patterns_path}")

# Continue with your network processing or analysis using the enriched dataset
# Define relative paths
node_metrics_files = {
    'Directed': os.path.join(base_dir, '../../data/output/node_metrics_patterns_directed.csv'),
    'Undirected': os.path.join(base_dir, '../../data/output/node_metrics_patterns_undirected.csv'),
    'Multigraph': os.path.join(base_dir, '../../data/output/node_metrics_patterns_multigraph.csv')
}
# Define paths to the graph files
directed_graph_path = os.path.join(base_dir, '../../data/output/initial_network_patterns_directed.csv')
undirected_graph_path = os.path.join(base_dir, '../../data/output/initial_network_patterns_undirected.csv')
multigraph_path = os.path.join(base_dir, '../../data/output/initial_multigraph_patterns.csv')

# Load graphs
def load_graph(path, graph_type):
    edges_df = pd.read_csv(path, delimiter=',', header=0)
    if graph_type == "Multigraph":
        edges_df['weight'] = pd.to_numeric(edges_df['weight'], errors='coerce')
        edges_df = edges_df.dropna(subset=['weight'])
        return nx.from_pandas_edgelist(edges_df, source='source', target='target', edge_attr='weight', create_using=nx.MultiGraph())
    elif graph_type == "Undirected":
        return nx.from_pandas_edgelist(edges_df, source='source', target='target', edge_attr='weight', create_using=nx.Graph())
    else:
        return nx.from_pandas_edgelist(edges_df, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

initial_directed = load_graph(directed_graph_path, "Directed")
initial_undirected = load_graph(undirected_graph_path, "Undirected")
initial_multigraph = load_graph(multigraph_path, "Multigraph")
# Load the CSV file (replace with the actual file path)
graph_path =  os.path.join(base_dir, '../../data/output/initial_network_patterns_directed.csv')
graph_df = pd.read_csv(graph_path)

# Group by source node and aggregate targets and weights
aggregated_patterns = (
    graph_df.groupby('source')
    .agg(
        targets=('target', lambda x: list(x)),  # Create a list of targets
        total_weight=('weight', 'sum')        # Sum the weights
    )
    .reset_index()
)

# Save the aggregated results to a new CSV file
aggregated_output_path = os.path.join(base_dir, '../../data/output/aggregated_patterns_directed.csv')
aggregated_patterns.to_csv(aggregated_output_path, index=False)
print(f"Aggregated patterns saved to {aggregated_output_path}")

# Display the first few rows for verification
print(aggregated_patterns.head())
# Load node metrics
node_metrics = []
for graph_type, file_path in node_metrics_files.items():
    df = pd.read_csv(file_path)
    df['graph_type'] = graph_type
    node_metrics.append(df)

node_metrics_df = pd.concat(node_metrics, ignore_index=True)
output_node_metrics_file = os.path.join(base_dir, '../../data/output/combined_node_metrics_patterns.csv')
node_metrics_df.to_csv(output_node_metrics_file, index=False)
print(f"Combined node metrics saved to {output_node_metrics_file}")

# Plot degree centrality
plt.figure(figsize=(12, 6))
sns.boxplot(x='graph_type', y='degree_centrality', data=node_metrics_df, palette='Set2', hue=None,legend=False)
plt.title('Degree Centrality Across Graph Types')
plt.xlabel('Graph Type')
plt.ylabel('Degree Centrality')
plt.tight_layout()
plt.show()

# Plot closeness centrality
plt.figure(figsize=(12, 6))
sns.boxplot(x='graph_type', y='closeness_centrality', data=node_metrics_df, palette='Set2', hue=None,legend=False)
plt.title('Closeness Centrality Across Graph Types')
plt.xlabel('Graph Type')
plt.ylabel('Closeness Centrality')
plt.tight_layout()
plt.show()

# Plot betweenness centrality
plt.figure(figsize=(12, 6))
sns.boxplot(x='graph_type', y='betweenness_centrality', data=node_metrics_df, palette='Set2', hue=None,legend=False)
plt.title('Betweenness Centrality Across Graph Types')
plt.xlabel('Graph Type')
plt.ylabel('Betweenness Centrality')
plt.tight_layout()
plt.show()

# Plot the distribution of patterns per performer-title combination
pattern_counts = merged_df.groupby(['performer', 'title']).size().reset_index(name='pattern_count')

# Save the distribution data
distribution_output_path = os.path.join(base_dir, '../../data/output/pattern_distribution.csv')
pattern_counts.to_csv(distribution_output_path, index=False)
print(f"Pattern distribution saved to {distribution_output_path}")

# Plot the distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=pattern_counts, x='pattern_count', bins=20, kde=True)
plt.title('Distribution of Patterns per Performer-Title Combination')
plt.xlabel('Number of Patterns')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
# Highlight top nodes in visualization
top_nodes_summary = {}
for graph_type in node_metrics_files.keys():
    top_nodes = (
        node_metrics_df[node_metrics_df['graph_type'] == graph_type]
        .nlargest(5, 'degree_centrality')[['node', 'degree_centrality']]
    )
    top_nodes_summary[graph_type] = top_nodes

# Combine top nodes for all graph types into a single DataFrame
top_nodes_df = pd.concat(
    [
        df.assign(
            graph_type=graph_type,
            degree=[
                initial_multigraph.degree(node) if graph_type == 'Multigraph'
                else initial_undirected.degree(node) if graph_type == 'Undirected'
                else initial_directed.in_degree(node) + initial_directed.out_degree(node)
                for node in df['node']
            ]
        )
        for graph_type, df in top_nodes_summary.items()
    ],
    ignore_index=True
)
top_nodes_output_file = os.path.join(base_dir, '../../data/output/top_nodes_by_degree_centrality_patterns.csv')
top_nodes_df.to_csv(top_nodes_output_file, index=False)
print(f"Top nodes by degree centrality saved to {top_nodes_output_file}")

# Define consistent layout for graphs
layout = nx.spring_layout(initial_directed, seed=42, k=0.9)

# Visualize graphs with unified colours and labels only for top nodes
from adjustText import adjust_text

fig, axes = plt.subplots(1, 3, figsize=(20, 9))

for ax, graph, graph_type, title in zip(
    axes,
    [initial_directed, initial_undirected, initial_multigraph],
    ['Directed', 'Undirected', 'Multigraph'],
    ['Directed Graph', 'Undirected Graph', 'Multigraph']
):
    top_nodes = top_nodes_summary[graph_type]['node'].values  # Get top nodes for the current graph type
    
    # Create labels for the nodes
    node_labels = {
        node: f"Pattern {node}" if node in top_nodes else ''
        for node in graph.nodes
    }

    # Adjust node sizes and colours
    node_colours = ['#f97c7c' if node in top_nodes else '#5588ff' for node in graph.nodes]
    node_sizes = [1000 * nx.degree_centrality(graph)[node] for node in graph.nodes]

    # Draw the graph
    nx.draw(
        graph,
        pos=layout,
        ax=ax,
        with_labels=False,  # Disable default labels
        node_color=node_colours,
        edge_color='gray',
        alpha=0.7,  # Consistent transparency
        node_size=node_sizes,
    )
    
    # Add labels manually to avoid overlaps
    texts = []
    for node in graph.nodes:
        if node in top_nodes:
            x, y = layout[node]
            label = node_labels[node]
            texts.append(ax.text(x, y, label, fontsize=10, ha='center', va='center', fontweight='bold'))

    # Adjust text positions to avoid overlap
    adjust_text(texts, ax=ax)

    ax.set_title(f"{title} (Top Nodes Highlighted)")

plt.tight_layout()
plt.show()

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/mcmc_sampling_solos.py
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

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/bayes/priors_matrix_solos.py
import numpy as np
import pandas as pd
import os

# Define relative paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data')
output_dir = os.path.join(data_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

prior_matrix_output_path = os.path.join(output_dir, 'prior_matrix_solos.npy')

# Load the DataFrame from the CSV file
solos_df = pd.read_csv(os.path.join(data_dir, 'solos_db_with_durations_cleaned.csv'))

# Extract relevant columns: melid, performer, title, release_date
solos_df = solos_df[['melid', 'performer', 'title', 'releasedate']]

# Sort the solos based on release date to maintain chronological order
solos_df.sort_values(by='releasedate', inplace=True)

# Create a dictionary to map melid to release date for easy lookup
melid_to_release_date = solos_df.set_index('melid')['releasedate'].to_dict()

# Create a list of unique melid values
valid_melids = solos_df['melid'].unique()
num_valid_solos = len(valid_melids)

# Create an empty prior matrix based on the number of valid solos
prior_matrix = np.zeros((num_valid_solos, num_valid_solos))

# Create a mapping from melid to index in the prior matrix
melid_to_index = {melid: i for i, melid in enumerate(valid_melids)}

# Iterate through each pair of melid values to determine temporal influence
for melid_a in valid_melids:
    release_date_a = melid_to_release_date[melid_a]

    for melid_b in valid_melids:
        release_date_b = melid_to_release_date[melid_b]

        # If melid_a has an earlier release date than melid_b, it can be an influence
        if release_date_a < release_date_b:
            prior_matrix[melid_to_index[melid_a], melid_to_index[melid_b]] = 1

print(f"Prior Matrix Shape: {prior_matrix.shape}")

# Save the prior matrix to a file
np.save(prior_matrix_output_path, prior_matrix)

print(f"Prior matrix saved to {prior_matrix_output_path}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/alignment/run_alignment.py
import subprocess

# Step 1: Prepare sequences
print("Preparing Mongeau-Sankoff compatible sequences...")
subprocess.run(["python", "src/preprocessing/MS_sequences_patterns.py"], check=True)
subprocess.run(["python", "src/preprocessing/MS_sequences_solos.py"], check=True)

# Step 2: Perform alignment
print("Running Mongeau-Sankoff alignment for patterns...")
subprocess.run(["python", "src/alignment/run_mongeau_sankoff_patterns.py"], check=True)

print("Running Mongeau-Sankoff alignment for solos...")
subprocess.run(["python", "src/alignment/run_mongeau_sankoff_solos.py"], check=True)

print("Alignment process completed!")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/alignment/run_mongeau_sankoff_solos.py
import sys
import os
import time
import numpy as np
import pandas as pd
import itertools
# Add directory to find function
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.alignment.mongeau_sankoff_functions import mongeau_sankoff_alignment

input_file_path = os.path.join(os.path.dirname(__file__), '../../data/sequences_solos.csv')


# Load the full solo dataset
sequences_df = pd.read_csv(input_file_path)
print(f"Loaded {len(sequences_df)} rows from the dataset.")

# Define alignment parameter of mongeau_sankoff_function
k2= 0.75

# Validate sequences format
sequences = sequences_df['sequence_tuples'].apply(eval)

# Sample of tuples to validate results
max_tuples = 50
subsampled_sequences = sequences.apply(lambda seq: seq[:max_tuples])

# Create combinations of the subsampled sequences along with their IDs for pairwise alignment
sequence_pairs = list(itertools.combinations(zip(subsampled_sequences, sequences_df['melid']), 2))

# Initiate empty to store results
aligned_results = []

# Store  maximum alignment score found
max_alignment_score = 0

# Total time for all pairs
total_run_start_time = time.time()

# Iteration through all sequence pairs for alignment to find the maximum alignment score
for index, ((seq1, seq1_id), (seq2, seq2_id)) in enumerate(sequence_pairs):  
    total_start_time = time.time()
    print(f"\nProcessing alignment for pair {index + 1} of {len(sequence_pairs)}")
    
    # Run the Mongeau-Sankoff alignment
    alignment_quality = mongeau_sankoff_alignment(seq1, seq2,k2)

    # Total time for processing the pair
    total_elapsed_time = time.time() - total_start_time

    # Print summarized timing information for each pair
    print(f"Alignment for pair {index + 1} completed in {total_elapsed_time:.2f} seconds")

    # Update the max alignment score found
    if alignment_quality > max_alignment_score:
        max_alignment_score = alignment_quality

    # Add the result along with sequence IDs for tracking
    aligned_results.append({
        'sequence_1_id': seq1_id,
        'sequence_2_id': seq2_id,
        'alignment_quality': alignment_quality
    })

# Calculate the total elapsed time for all pairs
total_run_elapsed_time = time.time() - total_run_start_time

# Print the total elapsed time
print(f"\nTotal time for processing all pairs: {total_run_elapsed_time:.2f} seconds")

# Convert and save normalised results into a DataFrame
aligned_results_df = pd.DataFrame(aligned_results)
aligned_results_df['similarity'] = aligned_results_df['alignment_quality'] / max_alignment_score
output_file_path = os.path.join(os.path.dirname(__file__), f'../../data/output/aligned_solos_results_k2_{k2:.2f}.csv')
aligned_results_df.to_csv(output_file_path, index=False)
print(f"Alignment results saved to {output_file_path}")

# Create similarity matrix 
unique_ids = sequences_df['melid'].unique()
similarity_matrix = pd.DataFrame(np.zeros((len(unique_ids), len(unique_ids))), index=unique_ids, columns=unique_ids)

# Fill similarity matrix
for _, row in aligned_results_df.iterrows():
    similarity_matrix.loc[row['sequence_1_id'], row['sequence_2_id']] = row['similarity']
    similarity_matrix.loc[row['sequence_2_id'], row['sequence_1_id']] = row['similarity']

# Fill diagonal with similarity = 1 for self-alignment
np.fill_diagonal(similarity_matrix.values, 1)

# Save the similarity matrix as CSV
similarity_matrix_file_path = os.path.join(os.path.dirname(__file__), f'../../data/output/similarity_matrix_solos_k2_{k2:.2f}.csv')
similarity_matrix.to_csv(similarity_matrix_file_path)
print(f"Similarity matrix saved to {similarity_matrix_file_path}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/alignment/MS_sequences_patterns.py
import pandas as pd
import os

# Define directory + Input/Output paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data')
output_path = os.path.join(data_dir, 'sequences_patterns.csv')
top_patterns_file_path = os.path.join(os.path.dirname(__file__), '../../data/top_patterns_sample.csv')

# Load data
matched_patterns = pd.read_csv(os.path.join(data_dir, 'matched_patterns_cleaned.csv'))
top_patterns_sample = pd.read_csv(top_patterns_file_path)

# Filter by columns needed
base_sequences = matched_patterns[['pattern_id', 'melid', 'pitch', 'sixteenth_representation', 'key', 'value']]
print(top_patterns_sample['pattern_id'].dtype)
print(base_sequences['pattern_id'].dtype)

# Filter base_sequences using top_patterns_sample
top_patterns_sample['pattern_id'] = top_patterns_sample['pattern_id'].astype(str)
base_sequences['pattern_id'] = base_sequences['pattern_id'].astype(str)

filtered_base_sequences = base_sequences[base_sequences['pattern_id'].isin(top_patterns_sample['pattern_id'])]
base_sequences = filtered_base_sequences

# Define a dictionary for tonic pitch classes (including modes and chromatic)
TONICS = {
    'C-maj': 60, 'C-min': 60, 'C-dor': 60, 'C-phry': 60, 'C-lyd': 60, 'C-mix': 60, 'C-aeo': 60, 'C-loc': 60, 'C-chrom': 60,
    'C#-maj': 61, 'C#-min': 61, 'C#-dor': 61, 'C#-phry': 61, 'C#-lyd': 61, 'C#-mix': 61, 'C#-aeo': 61, 'C#-loc': 61, 'C#-chrom': 61,
    'Db-maj': 61, 'Db-min': 61, 'Db-dor': 61, 'Db-phry': 61, 'Db-lyd': 61, 'Db-mix': 61, 'Db-aeo': 61, 'Db-loc': 61, 'Db-chrom': 61,
    'D-maj': 62, 'D-min': 62, 'D-dor': 62, 'D-phry': 62, 'D-lyd': 62, 'D-mix': 62, 'D-aeo': 62, 'D-loc': 62, 'D-chrom': 62,
    'Eb-maj': 63, 'Eb-min': 63, 'Eb-dor': 63, 'Eb-phry': 63, 'Eb-lyd': 63, 'Eb-mix': 63, 'Eb-aeo': 63, 'Eb-loc': 63, 'Eb-chrom': 63,
    'E-maj': 64, 'E-min': 64, 'E-dor': 64, 'E-phry': 64, 'E-lyd': 64, 'E-mix': 64, 'E-aeo': 64, 'E-loc': 64, 'E-chrom': 64,
    'F-maj': 65, 'F-min': 65, 'F-dor': 65, 'F-phry': 65, 'F-lyd': 65, 'F-mix': 65, 'F-aeo': 65, 'F-loc': 65, 'F-chrom': 65,
    'F#-maj': 66, 'F#-min': 66, 'F#-dor': 66, 'F#-phry': 66, 'F#-lyd': 66, 'F#-mix': 66, 'F#-aeo': 66, 'F#-loc': 66, 'F#-chrom': 66,
    'Gb-maj': 66, 'Gb-min': 66, 'Gb-dor': 66, 'Gb-phry': 66, 'Gb-lyd': 66, 'Gb-mix': 66, 'Gb-aeo': 66, 'Gb-loc': 66, 'Gb-chrom': 66,
    'G-maj': 67, 'G-min': 67, 'G-dor': 67, 'G-phry': 67, 'G-lyd': 67, 'G-mix': 67, 'G-aeo': 67, 'G-loc': 67, 'G-chrom': 67, 'Ab':68,
    'Ab-maj': 68, 'Ab-min': 68, 'Ab-dor': 68, 'Ab-phry': 68, 'Ab-lyd': 68, 'Ab-mix': 68, 'Ab-aeo': 68, 'Ab-loc': 68, 'Ab-chrom': 68,
    'A-maj': 69, 'A-min': 69, 'A-dor': 69, 'A-phry': 69, 'A-lyd': 69, 'A-mix': 69, 'A-aeo': 69, 'A-loc': 69, 'A-chrom': 69,
    'Bb-maj': 70, 'Bb-min': 70, 'Bb-dor': 70, 'Bb-phry': 70, 'Bb-lyd': 70, 'Bb-mix': 70, 'Bb-aeo': 70, 'Bb-loc': 70, 'Bb-chrom': 70,
    'B-maj': 71, 'B-min': 71, 'B-dor': 71, 'B-phry': 71, 'B-lyd': 71, 'B-mix': 71, 'B-aeo': 71, 'B-loc': 71, 'B-chrom': 71
}

# Function to calculate distance from pitch to the tonic
def calculate_intervals_from_tonic(row):
    tonic = TONICS[row['key']] 
    if isinstance(row['pitch'], list):  
        intervals = [(pitch - tonic) % 12 for pitch in row['pitch']]
    else:  # Handle single pitch
        intervals = (row['pitch'] - tonic) % 12
    
    # Adjust intervals for the shortest path after applying mmodulus 12
    if isinstance(intervals, list):
        intervals = [interval - 12 if interval > 6 else interval for interval in intervals]
    else:
        intervals = intervals - 12 if intervals > 6 else intervals
    
    return intervals

# Validate formats
base_sequences['pitch'] = pd.to_numeric(base_sequences['pitch'], errors='coerce').astype(int)
base_sequences['sixteenth_representation'] = pd.to_numeric(base_sequences['sixteenth_representation'], errors='coerce').astype(int)

# Apply function
base_sequences['interval'] = base_sequences.apply(calculate_intervals_from_tonic, axis=1)

# Group by 'pattern_id' to create sequences as tuples
def create_pattern_sequence(group):
    # Generate a list of (interval, sixteenth_representation) tuples for each pattern
    return [(int(interval), int(sixteenth)) for interval, sixteenth in zip(group['interval'], group['sixteenth_representation'])]

# Create sequences by grouping by 'pattern_id' and applying the function
sequences = (
    base_sequences.groupby('pattern_id')[['interval', 'sixteenth_representation']]
    .apply(create_pattern_sequence)
    .reset_index()
)

sequences.rename(columns={0: 'sequence_tuples'}, inplace=True)

# Merge the generated sequences with unique metadata for future analysis
metadata_columns = ['pattern_id', 'melid', 'key', 'value']
sequences = sequences.merge(base_sequences[metadata_columns].drop_duplicates(), on='pattern_id')

# Save the sequences into a CSV for further use
sequences.to_csv(output_path, index=False)
print(f"Sequences with intervals saved to {output_path}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/alignment/pattern_alignment_val.py
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define the directory and file pattern for the similarity matrices
current_dir = os.path.dirname(__file__)
directory = os.path.join(current_dir, '../../data/output/')
file_similarity = 'similarity_matrix_patterns_k2_'

# List of k2 values
k2_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50,0.75]

# Initialize a list to store results
results = []

for k2 in k2_values:
    # Construct the file path
    file_path = os.path.join(directory, f'{file_similarity}{k2:.2f}.csv')
    
    try:
        # Load the similarity matrix
        similarity_matrix = pd.read_csv(file_path, index_col=0)
    
        
        # Ensure the similarity matrix is square
        similarity_matrix.columns = similarity_matrix.columns.astype(int)
        common_ids = similarity_matrix.index.intersection(similarity_matrix.columns)
        similarity_matrix = similarity_matrix.loc[common_ids, common_ids]
        print(f"Index IDs: {similarity_matrix.index}")
        print(f"Column IDs: {similarity_matrix.columns}")
        if similarity_matrix.empty:
            raise ValueError("Similarity matrix is empty after filtering")
        
        # Handle missing values
        if similarity_matrix.isnull().values.any():
            print(f"Warning: Missing values found in the similarity matrix for k2={k2}")
            similarity_matrix.fillna(0, inplace=True)
        # Define a similarity threshold (e.g., 0.5)
        similarity_threshold = 0.5

        # Apply threshold to similarity matrix
        similarity_matrix = similarity_matrix.where(similarity_matrix >= similarity_threshold, other=0)
        # Convert to a graph
        graph = nx.from_pandas_adjacency(similarity_matrix)
        
        # Calculate metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        avg_clustering = nx.average_clustering(graph)
        
        # Degree centrality statistics
        degree_centrality = nx.degree_centrality(graph)
        centrality_values = list(degree_centrality.values())
        avg_degree_centrality = sum(centrality_values) / len(centrality_values)
        max_degree_centrality = max(centrality_values)
        median_degree_centrality = sorted(centrality_values)[len(centrality_values) // 2]
        
        # Store the results
        results.append({
            'k2': k2,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_clustering': avg_clustering,
            'avg_degree_centrality': avg_degree_centrality,
            'max_degree_centrality': max_degree_centrality,
            'median_degree_centrality': median_degree_centrality
        })
    
    except Exception as e:
        print(f"Failed to process file for k2={k2}: {e}")

# Convert results to a DataFrame
metrics_df = pd.DataFrame(results)

# Save the results to a CSV file
output_file_path = os.path.join(directory, 'alignment_metrics_patterns.csv')
metrics_df.to_csv(output_file_path, index=False)

print(f"Metrics saved to {output_file_path}")
print(metrics_df)


file_alignment = 'aligned_patterns_results_k2_'

# Initialize a list to store results
summary_results = []

for k2 in k2_values:
    # Construct the file path
    file_path = os.path.join(directory, f'{file_alignment}{k2:.2f}.csv')
    
    try:
        # Load the alignment results
        alignment_results = pd.read_csv(file_path)
        
        # Ensure the necessary column exists
        if 'alignment_quality' not in alignment_results.columns:
            raise ValueError(f"File {file_path} does not contain the column 'alignment_quality'")
        
        # Compute statistics
        stats = {
            'k2': k2,
            'mean': alignment_results['alignment_quality'].mean(),
            'median': alignment_results['alignment_quality'].median(),
            'std_dev': alignment_results['alignment_quality'].std(),
            'min': alignment_results['alignment_quality'].min(),
            'max': alignment_results['alignment_quality'].max(),
            '25th_percentile': alignment_results['alignment_quality'].quantile(0.25),
            '75th_percentile': alignment_results['alignment_quality'].quantile(0.75)
        }
        
        summary_results.append(stats)
        
        # Plot histogram and boxplot
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(alignment_results['alignment_quality'], bins=30, alpha=0.7, color='blue')
        plt.title(f'Alignment Quality Distribution (k2={k2:.2f})')
        plt.xlabel('Alignment Quality')
        plt.ylabel('Frequency')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(alignment_results['alignment_quality'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        plt.title(f'Alignment Quality Boxplot (k2={k2:.2f})')
        plt.xlabel('Alignment Quality')
        
        # Save the plot
        output_plot_path = os.path.join(directory, f'alignment_quality_analysis_k2_{k2:.2f}.png')
        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()
        
        print(f"Plots for k2={k2} saved to {output_plot_path}")
    
    except Exception as e:
        print(f"Error processing file for k2={k2}: {e}")

# Convert the summary results to a DataFrame
summary_df = pd.DataFrame(summary_results)

# Save the summary statistics to a CSV file
summary_file_path = os.path.join(directory, 'alignment_summary_patterns.csv')
summary_df.to_csv(summary_file_path, index=False)

print(f"Summary statistics saved to {summary_file_path}")
print(summary_df)

plt.figure(figsize=(10, 6))
plt.plot(summary_df['k2'], summary_df['mean'], label='Mean', marker='o')
plt.plot(summary_df['k2'], summary_df['median'], label='Median', marker='o')
plt.fill_between(summary_df['k2'], 
                 summary_df['mean'] - summary_df['std_dev'], 
                 summary_df['mean'] + summary_df['std_dev'], 
                 color='blue', alpha=0.2, label='1 Std Dev Range')
plt.title('Alignment Quality Across k2 Values')
plt.xlabel('k2')
plt.ylabel('Alignment Quality')
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(summary_df['k2'], summary_df['25th_percentile'], label='25th Percentile', marker='o')
plt.plot(summary_df['k2'], summary_df['75th_percentile'], label='75th Percentile', marker='o')
plt.fill_between(summary_df['k2'], 
                 summary_df['25th_percentile'], 
                 summary_df['75th_percentile'], 
                 color='green', alpha=0.2, label='Interquartile Range')
plt.title('Percentile Analysis of Alignment Quality')
plt.xlabel('k2')
plt.ylabel('Alignment Quality')
plt.legend()
plt.show()

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/alignment/run_mongeau_sankoff_patterns.py
import pandas as pd
import itertools
import time
import os
import sys
import numpy as np

# Add directory to find function
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.alignment.mongeau_sankoff_functions import mongeau_sankoff_alignment

# Define alignment parameter of mongeau_sankoff_function
k2 = 0.75 

# Input and output paths
sequences_file_path = os.path.join(os.path.dirname(__file__), '../../data/sequences_patterns.csv')
output_dir = os.path.join(os.path.dirname(__file__), '../../data/output/')
os.makedirs(output_dir, exist_ok=True)

# Load the sequences dataset
sequences_df = pd.read_csv(sequences_file_path)

# Step 1: Validate structure of input data
if 'pattern_id' not in sequences_df.columns or 'sequence_tuples' not in sequences_df.columns:
    raise ValueError("Input file must contain 'pattern_id' and 'sequence_tuples' columns.")

try:
    sequences_df['sequence_tuples'] = sequences_df['sequence_tuples'].apply(eval)
except Exception as e:
    raise ValueError(f"Error parsing sequence tuples: {e}")

# Check for duplicate pattern IDs
if sequences_df['pattern_id'].duplicated().any():
    raise ValueError("Duplicate pattern IDs detected in input data.")

print(f"Loaded {len(sequences_df)} patterns from sequences_patterns.csv")
print(f"Unique pattern IDs: {sequences_df['pattern_id'].nunique()}")

# Drop rows with null sequences
if sequences_df['sequence_tuples'].isnull().any():
    print("Warning: Null sequences detected. These will be removed.")
    sequences_df = sequences_df.dropna(subset=['sequence_tuples'])

# Step 2: Create combinations of the sequences for pairwise alignment
sequence_pairs = list(itertools.combinations(zip(sequences_df['sequence_tuples'], sequences_df['pattern_id']), 2))
print(f"Generated {len(sequence_pairs)} sequence pairs for alignment.")

aligned_results = []  # To store results
max_alignment_score = 0  # Track the max alignment score

total_run_start_time = time.time()

# Step 3: Process each pair
for index, ((seq1, seq1_id), (seq2, seq2_id)) in enumerate(sequence_pairs):
    print(f"Processing alignment for pair {index + 1} of {len(sequence_pairs)}")

    start_time = time.time()
    try:
        alignment_quality = mongeau_sankoff_alignment(seq1, seq2, k2)
    except Exception as e:
        print(f"Error processing pair {seq1_id}-{seq2_id}: {e}")
        continue
    
    elapsed_time = time.time() - start_time
    print(f"Processed pair {index + 1} in {elapsed_time:.2f} seconds")
    
    max_alignment_score = max(max_alignment_score, alignment_quality)

    aligned_results.append({
        'pattern_id_1': seq1_id,
        'pattern_id_2': seq2_id,
        'alignment_quality': alignment_quality
    })

# Validate alignment results
if not aligned_results:
    raise ValueError("No alignment results generated. Check input data and alignment function.")

aligned_results_df = pd.DataFrame(aligned_results)

# Step 4: Normalise and save alignment results
aligned_results_df['similarity'] = aligned_results_df['alignment_quality'] / max_alignment_score
output_file_path = os.path.join(output_dir, f'aligned_patterns_results_k2_{k2:.2f}.csv')
aligned_results_df.to_csv(output_file_path, index=False)
print(f"Alignment results saved to {output_file_path}")

# Validate coverage of pattern IDs in results
result_pattern_ids = set(aligned_results_df['pattern_id_1']).union(set(aligned_results_df['pattern_id_2']))
missing_pattern_ids = set(sequences_df['pattern_id']) - result_pattern_ids
if missing_pattern_ids:
    print(f"Warning: Missing pattern IDs in alignment results: {missing_pattern_ids}")

# Step 5: Create and validate the similarity matrix
unique_ids = sequences_df['pattern_id'].unique()
similarity_matrix = pd.DataFrame(np.zeros((len(unique_ids), len(unique_ids))), index=unique_ids, columns=unique_ids)

# Populate similarity matrix
for _, row in aligned_results_df.iterrows():
    similarity_matrix.loc[row['pattern_id_1'], row['pattern_id_2']] = row['similarity']
    similarity_matrix.loc[row['pattern_id_2'], row['pattern_id_1']] = row['similarity']

np.fill_diagonal(similarity_matrix.values, 1)  # Self-alignment is 1

# Check for null values in the matrix
if similarity_matrix.isnull().values.any():
    print("Warning: Similarity matrix contains null values.")
    print(similarity_matrix)

similarity_matrix_file_path = os.path.join(output_dir, f'similarity_matrix_patterns_k2_{k2:.2f}.csv')
similarity_matrix.to_csv(similarity_matrix_file_path)
print(f"Similarity matrix saved to {similarity_matrix_file_path}")

# Step 6: Validate similarity matrix dimensions
if similarity_matrix.shape[0] != len(unique_ids) or similarity_matrix.shape[1] != len(unique_ids):
    raise ValueError("Mismatch in similarity matrix dimensions.")

print("Alignment process completed successfully.")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/alignment/MS_sequences_solos.py
import pandas as pd
import ast
import os

# Define directory + Input/Output paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data')
output_path = os.path.join(data_dir, 'sequences_solos.csv')

# Load data
solos_data = pd.read_csv(os.path.join(data_dir, 'solos_db_with_durations_cleaned.csv'))

# Filter by columns needed
full_solos_sequences = solos_data[['melid', 'pitch', 'calculated_duration', 'key']].copy()

# Helper function evaluate pitch
def safe_eval_pitch(pitch):
    try:
        if isinstance(pitch, str):
            return [float(p) for p in ast.literal_eval(pitch)]
        elif isinstance(pitch, (float, int)):
            return [float(pitch)]
        elif isinstance(pitch, list):
            return [float(p) for p in pitch]
        else:
            return []
    except Exception as e:
        print(f"Error parsing pitch: {pitch}. Error: {e}")
        return []

# Apply evaluation function
full_solos_sequences['pitch'] = full_solos_sequences['pitch'].apply(safe_eval_pitch)

# Define a dictionary for tonic pitch classes (including modes and chromatic)
TONICS = {
    'C-maj': 60, 'C-min': 60, 'C-dor': 60, 'C-phry': 60, 'C-lyd': 60, 'C-mix': 60, 'C-aeo': 60, 'C-loc': 60, 'C-chrom': 60, 'C-blues': 60,
    'C#-maj': 61, 'C#-min': 61, 'C#-dor': 61, 'C#-phry': 61, 'C#-lyd': 61, 'C#-mix': 61, 'C#-aeo': 61, 'C#-loc': 61, 'C#-chrom': 61, 'C#-blues': 61,
    'Db-maj': 61, 'Db-min': 61, 'Db-dor': 61, 'Db-phry': 61, 'Db-lyd': 61, 'Db-mix': 61, 'Db-aeo': 61, 'Db-loc': 61, 'Db-chrom': 61, 'Db-blues': 61,
    'D-maj': 62, 'D-min': 62, 'D-dor': 62, 'D-phry': 62, 'D-lyd': 62, 'D-mix': 62, 'D-aeo': 62, 'D-loc': 62, 'D-chrom': 62, 'D-blues': 62,
    'D#-maj': 63, 'D#-min': 63, 'D#-dor': 63, 'D#-phry': 63, 'D#-lyd': 63, 'D#-mix': 63, 'D#-aeo': 63, 'D#-loc': 63, 'D#-chrom': 63, 'D#-blues': 63,
    'Eb-maj': 63, 'Eb-min': 63, 'Eb-dor': 63, 'Eb-phry': 63, 'Eb-lyd': 63, 'Eb-mix': 63, 'Eb-aeo': 63, 'Eb-loc': 63, 'Eb-chrom': 63, 'Eb-blues': 63,
    'E-maj': 64, 'E-min': 64, 'E-dor': 64, 'E-phry': 64, 'E-lyd': 64, 'E-mix': 64, 'E-aeo': 64, 'E-loc': 64, 'E-chrom': 64, 'E-blues': 64,
    'F-maj': 65, 'F-min': 65, 'F-dor': 65, 'F-phry': 65, 'F-lyd': 65, 'F-mix': 65, 'F-aeo': 65, 'F-loc': 65, 'F-chrom': 65, 'F-blues': 65,
    'F#-maj': 66, 'F#-min': 66, 'F#-dor': 66, 'F#-phry': 66, 'F#-lyd': 66, 'F#-mix': 66, 'F#-aeo': 66, 'F#-loc': 66, 'F#-chrom': 66, 'F#-blues': 66,
    'Gb-maj': 66, 'Gb-min': 66, 'Gb-dor': 66, 'Gb-phry': 66, 'Gb-lyd': 66, 'Gb-mix': 66, 'Gb-aeo': 66, 'Gb-loc': 66, 'Gb-chrom': 66, 'Gb-blues': 66,
    'G-maj': 67, 'G-min': 67, 'G-dor': 67, 'G-phry': 67, 'G-lyd': 67, 'G-mix': 67, 'G-aeo': 67, 'G-loc': 67, 'G-chrom': 67, 'G-blues': 67,
    'Ab-maj': 68, 'Ab-min': 68, 'Ab-dor': 68, 'Ab-phry': 68, 'Ab-lyd': 68, 'Ab-mix': 68, 'Ab-aeo': 68, 'Ab-loc': 68, 'Ab-chrom': 68, 'Ab-blues': 68, 'Ab': 68,
    'A-maj': 69, 'A-min': 69, 'A-dor': 69, 'A-phry': 69, 'A-lyd': 69, 'A-mix': 69, 'A-aeo': 69, 'A-loc': 69, 'A-chrom': 69, 'A-blues': 69,
    'Bb-maj': 70, 'Bb-min': 70, 'Bb-dor': 70, 'Bb-phry': 70, 'Bb-lyd': 70, 'Bb-mix': 70, 'Bb-aeo': 70, 'Bb-loc': 70, 'Bb-chrom': 70, 'Bb-blues': 70,
    'B-maj': 71, 'B-min': 71, 'B-dor': 71, 'B-phry': 71, 'B-lyd': 71, 'B-mix': 71, 'B-aeo': 71, 'B-loc': 71, 'B-chrom': 71, 'B-blues': 71,
}

# Function to calculate distance from pitch to the tonic
def calculate_intervals_from_tonic(row):
    tonic = TONICS[row['key']]  # Assume the key is valid and exists in the dictionary
    if isinstance(row['pitch'], list):  # Handle lists of pitches
        intervals = [(pitch - tonic) % 12 for pitch in row['pitch']]
    else:  # Handle single pitch
        intervals = (row['pitch'] - tonic) % 12
    
    # Adjust intervals for the shortest path after applying mmodulus 12
    if isinstance(intervals, list):
        intervals = [interval - 12 if interval > 6 else interval for interval in intervals]
    else:
        intervals = intervals - 12 if intervals > 6 else intervals
    
    return intervals

# Apply function
full_solos_sequences['intervals'] = full_solos_sequences.apply(calculate_intervals_from_tonic, axis=1)

# Convert duration to sixteenths and validate format
full_solos_sequences['sixteenth_duration'] = full_solos_sequences['calculated_duration'].apply(
    lambda x: int(round(16 * x)) if pd.notnull(x) else 0
)

# Explode the rows to have one note per row
exploded_sequences = full_solos_sequences.explode(['pitch', 'intervals', 'sixteenth_duration'])

# Ensure numeric types
exploded_sequences['intervals'] = pd.to_numeric(exploded_sequences['intervals'], errors='coerce').fillna(0).astype(float)
exploded_sequences['sixteenth_duration'] = pd.to_numeric(exploded_sequences['sixteenth_duration'], errors='coerce').fillna(0).astype(int)

# Group by melid to create sequences of tuples
grouped_sequences = exploded_sequences.groupby('melid')[['intervals', 'sixteenth_duration']].apply(
    lambda x: [(int(interval), int(duration)) for interval, duration in zip(x['intervals'], x['sixteenth_duration'])]
).reset_index(name='sequence_tuples')

# Save sequences to CSV
grouped_sequences.to_csv(output_path, index=False)
print(f"Processed sequences saved to {output_path}")


# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/alignment/solos_alignment_val.py
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define the directory and file pattern for the similarity matrices
current_dir = os.path.dirname(__file__)
directory = os.path.join(current_dir, '../../data/output/')
file_similarity = 'similarity_matrix_solos_k2_'

# List of k2 values
k2_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75]

# Initialize a list to store results
results = []

for k2 in k2_values:
    # Construct the file path
    file_path = os.path.join(directory, f'{file_similarity}{k2:.2f}.csv')
    
    try:
        # Load the similarity matrix
        similarity_matrix = pd.read_csv(file_path, index_col=0)
        
        # Ensure the similarity matrix is square
        similarity_matrix.columns = similarity_matrix.columns.astype(int)
        common_ids = similarity_matrix.index.intersection(similarity_matrix.columns)
        similarity_matrix = similarity_matrix.loc[common_ids, common_ids]
        print(f"Index IDs: {similarity_matrix.index}")
        print(f"Column IDs: {similarity_matrix.columns}")
        if similarity_matrix.empty:
            raise ValueError("Similarity matrix is empty after filtering")
        
        # Handle missing values
        if similarity_matrix.isnull().values.any():
            print(f"Warning: Missing values found in the similarity matrix for k2={k2}")
            similarity_matrix.fillna(0, inplace=True)
        # Define a similarity threshold (e.g., 0.5)
        similarity_threshold = 0.5

        # Apply threshold to similarity matrix
        similarity_matrix = similarity_matrix.where(similarity_matrix >= similarity_threshold, other=0)
        # Convert to a graph
        graph = nx.from_pandas_adjacency(similarity_matrix)
        
        # Calculate metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        avg_clustering = nx.average_clustering(graph)
        
        # Degree centrality statistics
        degree_centrality = nx.degree_centrality(graph)
        centrality_values = list(degree_centrality.values())
        avg_degree_centrality = sum(centrality_values) / len(centrality_values)
        max_degree_centrality = max(centrality_values)
        median_degree_centrality = sorted(centrality_values)[len(centrality_values) // 2]
        
        # Store the results
        results.append({
            'k2': k2,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_clustering': avg_clustering,
            'avg_degree_centrality': avg_degree_centrality,
            'max_degree_centrality': max_degree_centrality,
            'median_degree_centrality': median_degree_centrality
        })
    
    except Exception as e:
        print(f"Failed to process file for k2={k2}: {e}")

# Convert results to a DataFrame
metrics_df = pd.DataFrame(results)

# Save the results to a CSV file
output_file_path = os.path.join(directory, 'alignment_metrics_solos.csv')
metrics_df.to_csv(output_file_path, index=False)

print(f"Metrics saved to {output_file_path}")
print(metrics_df)


file_alignment = 'aligned_solos_results_k2_'

# Initialize a list to store results
summary_results = []

for k2 in k2_values:
    # Construct the file path
    file_path = os.path.join(directory, f'{file_alignment}{k2:.2f}.csv')
    
    try:
        # Load the alignment results
        alignment_results = pd.read_csv(file_path)
        
        # Ensure the necessary column exists
        if 'alignment_quality' not in alignment_results.columns:
            raise ValueError(f"File {file_path} does not contain the column 'alignment_quality'")
        
        # Compute statistics
        stats = {
            'k2': k2,
            'mean': alignment_results['alignment_quality'].mean(),
            'median': alignment_results['alignment_quality'].median(),
            'std_dev': alignment_results['alignment_quality'].std(),
            'min': alignment_results['alignment_quality'].min(),
            'max': alignment_results['alignment_quality'].max(),
            '25th_percentile': alignment_results['alignment_quality'].quantile(0.25),
            '75th_percentile': alignment_results['alignment_quality'].quantile(0.75)
        }
        
        summary_results.append(stats)
        
        # Plot histogram and boxplot
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(alignment_results['alignment_quality'], bins=30, alpha=0.7, color='blue')
        plt.title(f'Alignment Quality Distribution (k2={k2:.2f})')
        plt.xlabel('Alignment Quality')
        plt.ylabel('Frequency')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(alignment_results['alignment_quality'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        plt.title(f'Alignment Quality Boxplot (k2={k2:.2f})')
        plt.xlabel('Alignment Quality')
        
        # Save the plot
        output_plot_path = os.path.join(directory, f'alignment_quality_analysis_solos_k2_{k2:.2f}.png')
        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()
        
        print(f"Plots for k2={k2} saved to {output_plot_path}")
    
    except Exception as e:
        print(f"Error processing file for k2={k2}: {e}")

# Convert the summary results to a DataFrame
summary_df = pd.DataFrame(summary_results)

# Save the summary statistics to a CSV file
summary_file_path = os.path.join(directory, 'alignment_summary_solos.csv')
summary_df.to_csv(summary_file_path, index=False)

print(f"Summary statistics saved to {summary_file_path}")
print(summary_df)

plt.figure(figsize=(10, 6))
plt.plot(summary_df['k2'], summary_df['mean'], label='Mean', marker='o')
plt.plot(summary_df['k2'], summary_df['median'], label='Median', marker='o')
plt.fill_between(summary_df['k2'], 
                 summary_df['mean'] - summary_df['std_dev'], 
                 summary_df['mean'] + summary_df['std_dev'], 
                 color='blue', alpha=0.2, label='1 Std Dev Range')
plt.title('Alignment Quality Across k2 Values')
plt.xlabel('k2')
plt.ylabel('Alignment Quality')
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(summary_df['k2'], summary_df['25th_percentile'], label='25th Percentile', marker='o')
plt.plot(summary_df['k2'], summary_df['75th_percentile'], label='75th Percentile', marker='o')
plt.fill_between(summary_df['k2'], 
                 summary_df['25th_percentile'], 
                 summary_df['75th_percentile'], 
                 color='green', alpha=0.2, label='Interquartile Range')
plt.title('Percentile Analysis of Alignment Quality')
plt.xlabel('k2')
plt.ylabel('Alignment Quality')
plt.legend()
plt.show()

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/alignment/mongeau_sankoff_functions.py
import numpy as np

# Mapping semitone differences to degrees
SEMITONE_TO_DEGREE = {
    0: 0,  # Unison
    1: 1,  # Minor second
    2: 2,  # Major second
    3: 2,  # Minor third
    4: 3,  # Major third
    5: 4,  # Perfect fourth
    6: 5,  # Tritone
    7: 4,  # Perfect fifth
    8: 5,  # Minor sixth
    9: 5,  # Major sixth
    10: 6, # Minor seventh
    11: 6  # Major seventh
}

# Mapping degrees to weights
DEGREE_WEIGHTS = {
    0: 0.0,   # Unison (identity replacement), octave, two octaves
    4: 0.1,   # Fifth, octave and a fifth
    2: 0.2,   # Third, octave and a third
    5: 0.35,  # Sixth
    3: 0.5,   # Fourth
    6: 0.8,   # Seventh
    1: 0.9    # Second
}

# Function to calculate degree weight based on interval
def get_degree_weight(interval):
    """
    Returns the weight for a given interval based on predefined degree weights.
    """
    # Step 1: Convert semitones to degrees
    degree = SEMITONE_TO_DEGREE.get(abs(interval) % 12, None)
    
    # Step 2: Get the weight for that degree, default to 2.6 for dissonant intervals if not found
    if degree is not None:
        return DEGREE_WEIGHTS.get(degree, 2.6)
    else:
        return 2.6

# Cost function for insertion (gaps)
def insertion_cost(note, k2):
    """Calculates the insertion cost for a given note."""
    return k2 * note[1]  # k2 * Length of inserted note (duration)

# Cost function for deletion
def deletion_cost(note, k2):
    """Calculates the deletion cost for a given note."""
    return k2 * note[1]  # k2 * Length of deleted note (duration)

# Cost function for substitution (replacement)
def substitution_cost(note1, note2, k2):
    """
    Calculates the substitution cost between two notes.
    The cost involves interval differences and duration differences.
    It is defined as k2 times the sum of the lengths of the two notes minus the weight of their interval differences and duration differences.

    Args:
    - note1: A tuple containing (pitch, duration) for the first note.
    - note2: A tuple containing (pitch, duration) for the second note.

    Returns:
    - The calculated substitution cost.
    """
    # Calculate interval weight based on pitch difference
    interval_degree_weight = get_degree_weight(note1[0] - note2[0])

    # Calculate interval cost as interval weight * shorter duration
    interval_weight = interval_degree_weight * min(note1[1], note2[1])

    # Calculate duration weight as absolute difference in duration
    duration_weight = abs(note1[1] - note2[1])

    # Total substitution cost
    return k2 * (note1[1] + note2[1]) - (interval_weight + duration_weight)

# Mongeau-Sankoff Alignment Function with Local Alignment
def mongeau_sankoff_alignment(sequence1, sequence2, k2):
    """
    Perform Mongeau-Sankoff alignment between two sequences.

    Args:
    - sequence1: List of tuples, where each tuple contains (pitch, duration) for sequence 1.
    - sequence2: List of tuples, where each tuple contains (pitch, duration) for sequence 2.

    Returns:
    - alignment_cost: The calculated alignment cost, representing the quality of the best alignment.
    """

    m, n = len(sequence1), len(sequence2)
    # Step 1: Initialize alignment matrix with zeros for local alignment
    alignment_matrix = np.zeros((m + 1, n + 1))
    
    max_value = 0

    # Step 2: Fill the alignment matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Consider different segment lengths for more complex substitution scoring
            score_sub = float('-inf')

            # Step 2.1: Segment scoring loop          
            for k in range(1, min(i, j) + 1):  # Loop over possible segment lengths
                segment_score = 0
                for s in range(k):
                    segment_score += substitution_cost(sequence1[i - s - 1], sequence2[j - s - 1], k2)
                score_sub_segment = alignment_matrix[i - k][j - k] + segment_score
                score_sub = max(score_sub, score_sub_segment)


            # Step 2.2: Calculate scores for deletion and insertion
            score_del = alignment_matrix[i - 1][j] + deletion_cost(sequence1[i - 1], k2)
            score_ins = alignment_matrix[i][j - 1] + insertion_cost(sequence2[j - 1], k2)

            # Step 2.3: Select the highest score or zero (for local alignment)
            alignment_matrix[i][j] = max(0, score_sub, score_del, score_ins)

            # Track the maximum score for traceback
            if alignment_matrix[i][j] > max_value:
                max_value = alignment_matrix[i][j]



    # Alignment quality is represented by the maximum value found for local alignment
    alignment_quality = max_value

    return alignment_quality

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/calculate_durations.py
import pandas as pd
import os

# Define relative paths for input and output
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')
input_file = os.path.join(data_dir, 'solos_db.csv')
output_file = os.path.join(data_dir, 'solos_db_with_durations.csv')

# Load the DataFrame from the CSV file (solos_db)
solos_db = pd.read_csv(input_file)

# Updated duration calculation function
def calculate_duration_with_tatum(row, next_row):
    # Base durations for each division type
    division_mapping = {
        1: 1 / 4,    # Quarter note
        2: 1 / 8,    # Eighth note
        3: 1 / 8,    # Approximate to eighth note for consistency with sixteenths
        4: 1 / 16,   # Sixteenth note
    }

    # For divisions 4 and above, default to sixteenth note
    if row['division'] >= 4:
        base_duration = 1 / 16
    else:
        # Get the base duration from the division mapping if it exists
        base_duration = division_mapping.get(row['division'], None)

    if base_duration is None:
        # If no valid base duration is found, return None
        return None

    # Calculate the duration based on the tatum positions
    if (
        next_row is not None and 
        row['bar'] == next_row['bar'] and
        row['beat'] == next_row['beat']
    ):
        tatum_current = row['tatum']
        tatum_next = next_row['tatum']
        tatum_difference = tatum_next - tatum_current
    else:
        # If it's the last note in the beat or the last note of the sequence
        tatum_difference = 1  # Assume it occupies a single tatum if it's the last one or no subsequent notes

    # Multiply base duration by tatum difference
    return base_duration * tatum_difference

# Function to convert durations into sixteenth-note representation
def convert_to_sixteenth_representation(duration):
    # Convert to sixteenth representation: 1/4 becomes 4, 1/8 becomes 2, 1/16 becomes 1
    if duration == 1 / 4:
        return 4
    elif duration == 1 / 8:
        return 2
    elif duration == 1 / 16:
        return 1
    else:
        # For other durations, calculate the equivalent "divided by 1/16"
        return int(duration / (1 / 16))

# Apply the function to the DataFrame
durations = []
sixteenth_representations = []

for i in range(len(solos_db)):
    current_row = solos_db.iloc[i]
    next_row = solos_db.iloc[i + 1] if i + 1 < len(solos_db) else None
    duration = calculate_duration_with_tatum(current_row, next_row)
    durations.append(duration)
    sixteenth_representations.append(convert_to_sixteenth_representation(duration) if duration is not None else None)

# Assign calculated durations back to the DataFrame
solos_db['calculated_duration'] = durations
solos_db['sixteenth_representation'] = sixteenth_representations

# Verify the output
print(solos_db[['bar', 'beat', 'tatum', 'division', 'calculated_duration', 'sixteenth_representation']].head(20))

# Export the updated DataFrame to a CSV file
solos_db.to_csv(output_file, index=False)
print("Durations added and saved to solos_db_with_durations.csv")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/patterns_top_sample.py
import pandas as pd
import os

# Define base and data directories using relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')

# Define file paths
input_file = os.path.join(data_dir, 'matched_patterns_cleaned.csv')
output_file = os.path.join(data_dir, 'top_patterns_sample.csv')

# Load the matched patterns dataset
matched_patterns_df = pd.read_csv(input_file)

# Create an empty DataFrame to store the top patterns for each solo
top_patterns_df = pd.DataFrame()

# Group by performer and title to extract the top 50 patterns by frequency for each combination
grouped = matched_patterns_df.groupby(['performer', 'title'])

for (performer, title), group in grouped:
    # Sort the group by frequency in descending order and take the top 50
    top_patterns = group.sort_values(by='pattern_frequency', ascending=False).head(50)
    # Append these top patterns to the resulting DataFrame
    top_patterns_df = pd.concat([top_patterns_df, top_patterns], ignore_index=True)

# Sort the resulting DataFrame by pattern_id and reset the index
top_patterns_df = top_patterns_df.sort_values(by='pattern_id').reset_index(drop=True)

# Save the resulting sample to a new CSV file
top_patterns_df.to_csv(output_file, index=False)

print(f"Representative sample saved to {output_file}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/solos_db_extraction.py
import pandas as pd
import sqlite3
import os

# Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')
db_file = os.path.join(data_dir, 'wjazzd.db')
output_file_path = os.path.join(data_dir, 'solos_db.csv')  # Final output path

# Specify the fields to extract from each table
fields_to_extract = {
    # Other tables remain unchanged
    "melody": [
        "eventid",
        "melid",
        "onset",
        "pitch",
        "duration",
        "period",
        "division",
        "bar",
        "beat",
        "tatum",
        "subtatum",
        "num",
        "denom",
        "beatprops",
        "beatdur",
        "tatumprops"
    ],
    "solo_info": [
        "melid",
        "trackid", 
        "compid", 
        "recordid", 
        "performer", 
        "title", 
        "solopart", 
        "instrument", 
        "style", 
        "avgtempo", 
        "tempoclass", 
        "rhythmfeel", 
        "key", 
        "signature", 
        "chord_changes", 
        "chorus_count"
    ],
    "record_info": ["recordid", "releasedate"],
    "composition_info": ["compid","composer", "form", "template", "tonalitytype", "genre"],
    "track_info": ["trackid","lineup"] 
}

def connect_to_database(db_path):
    """Connect to SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        exit(1)

def extract_table(conn, table_name, fields=None):
    """Extract data from a specific table."""
    query = f"SELECT {', '.join(fields) if fields else '*'} FROM {table_name}"
    try:
        df = pd.read_sql_query(query, conn)
        print(f"Successfully extracted {len(df)} rows from '{table_name}' table.")
        return df
    except sqlite3.Error as e:
        print(f"Error reading table '{table_name}': {e}")
        return None

# Connect to the database
conn = connect_to_database(db_file)

# Extract the relevant tables with selected fields
extracted_tables = {
    table: extract_table(conn, table, fields)
    for table, fields in fields_to_extract.items()
}

# Close the connection
conn.close()

# Merge the tables
# Step 1: Merge melody with solo_info
merged_df_1 = extracted_tables['melody'].merge(extracted_tables['solo_info'], on='melid', how='left')

# Step 2: Merge with record_info
merged_df_2 = merged_df_1.merge(extracted_tables['record_info'], on='recordid', how='left')

# Step 3: Merge with composition_info
merged_df_3 = merged_df_2.merge(extracted_tables['composition_info'], on='compid', how='left')

# (Optional) Step 4: Merge with track_info if needed
if 'track_info' in extracted_tables:
    merged_df_final = merged_df_3.merge(extracted_tables['track_info'], on='trackid', how='left')
else:
    merged_df_final = merged_df_3

# Drop duplicated columns if they exist
merged_df_final = merged_df_final.loc[:, ~merged_df_final.columns.duplicated()]

# Save the final DataFrame to CSV
merged_df_final.to_csv(output_file_path, index=False)
print(f"Final merged data saved to {output_file_path}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/pattern_extraction.py
import os
import subprocess
import pandas as pd

# Set the base directory to the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the input file
input_file_path = os.path.join(BASE_DIR, '..', 'data', 'solos_db_with_durations.csv')

# Validate input file
if not os.path.exists(input_file_path):
    print(f"Error: Input file does not exist at {input_file_path}")
    exit(1)

# Load the DataFrame from the CSV file
try:
    solos_full = pd.read_csv(input_file_path)
    print("First 10 rows of input file:")
    print(solos_full.head(10))
except Exception as e:
    print(f"Error reading input file: {e}")
    exit(1)

# Paths for the melpat executable and configuration file
melpat_path = '/Users/paola_amigo/Desktop/Thesis/melospy-suite_V_1_6_mac_osx/bin/melpat'
config_path = '/Users/paola_amigo/Desktop/Thesis/JazzSolos/melpat_config.yaml'

# Validate melpat executable
if not os.path.exists(melpat_path):
    print(f"Error: Melpat executable does not exist at {melpat_path}")
    exit(1)

# Validate configuration file
if not os.path.exists(config_path):
    print(f"Error: Configuration file does not exist at {config_path}")
    exit(1)

# Run the melpat command
try:
    command = [
        melpat_path,
        '-c', config_path,  # Specify the configuration file
        '--verbose'
    ]

    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    # Log success
    print("Pattern extraction successful!")
    print("Output from melpat:")
    print(result.stdout)

    # Optionally, write output to a log file
    output_log_path = os.path.join(BASE_DIR, '..', 'output', 'melpat_output.log')
    with open(output_log_path, 'w') as log_file:
        log_file.write(result.stdout)

except subprocess.CalledProcessError as e:
    print(f"Error running melpat: {e}")
    print("Command output:")
    print(e.stderr)

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/patterns_clean.py
import pandas as pd
import os

# Define base and data directories using relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')

# Define file paths
input_file = os.path.join(data_dir, 'matched_patterns.csv')
output_file = os.path.join(data_dir, 'matched_patterns_cleaned.csv')

# Load the matched patterns dataset
matched_patterns_df = pd.read_csv(input_file)

# Drop rows with missing 'key'
matched_patterns_df = matched_patterns_df.dropna(subset=['key'])

# Sort by pattern_id to ensure we keep the first occurrence in the case of duplicates
matched_patterns_df = matched_patterns_df.sort_values(by=['pattern_id'])

# Identify the first pattern_id for each unique combination of performer, title, and value
first_occurrences = matched_patterns_df.drop_duplicates(subset=['performer', 'title', 'value'], keep='first')

# Extract the pattern_ids that we want to keep
valid_pattern_ids = first_occurrences['pattern_id'].unique()

# Keep only rows with the selected pattern_ids, ensuring all rows of the first occurrence are kept
matched_patterns_df = matched_patterns_df[matched_patterns_df['pattern_id'].isin(valid_pattern_ids)]

# Calculate the frequency of each unique pattern value across different pattern_ids
# Group by 'value' and count the number of unique 'pattern_id' for each 'value'
pattern_frequency = matched_patterns_df.groupby('value')['pattern_id'].nunique()

# Map the calculated pattern frequency back to the DataFrame
matched_patterns_df['pattern_frequency'] = matched_patterns_df['value'].map(pattern_frequency)

# Save the cleaned dataset to a new CSV file
matched_patterns_df.to_csv(output_file, index=False)

print(f"Cleaned patterns saved to {output_file}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/run_preprocessing.py
import subprocess
import os

# Define the base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# List of scripts to execute in order
scripts = [
    "src/preprocessing/solos_db_extraction.py",
    "src/preprocessing/calculate_durations.py",
    "src/preprocessing/solos_db_clean.py",
    "src/preprocessing/pattern_extraction.py",
    "src/preprocessing/patterns_processed.py",
    "src/preprocessing/matched_patterns.py",
    "src/preprocessing/patterns_clean.py",
    "src/preprocessing/patterns_top_sample.py"
]

# Execute each script in order
for script in scripts:
    script_path = os.path.join(BASE_DIR, script)
    print(f"Running {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error occurred in {script_path}:\n{result.stderr}")
        break
    else:
        print(f"Finished {script_path}:\n{result.stdout}")

print("Preprocessing completed successfully!")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/solos_db_clean.py
import pandas as pd
import os
import re

# Define relative paths for input and output
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')
input_file = os.path.join(data_dir, 'solos_db_with_durations.csv')
output_file = os.path.join(data_dir, 'solos_db_with_durations_cleaned.csv')

# Read the CSV file
df = pd.read_csv(input_file)

print(f"Initial dataset size: {df.shape}")

# Step 1: Remove solos with missing 'releasedate'
df = df.dropna(subset=['releasedate'])
print(f"After removing missing 'releasedate': {df.shape}")

# Step 2: Handle duplicates by 'performer' and 'title' based on `melid` (keep the one with the fewest notes)
df['num_notes'] = df.groupby('melid')['eventid'].transform('count')
duplicates_by_performer = df[df.duplicated(subset=['performer', 'title'], keep=False)]
melids_to_keep = duplicates_by_performer.loc[
    duplicates_by_performer.groupby(['performer', 'title'])['num_notes'].idxmin()
]['melid'].unique()
filtered_performer_duplicates = df[df['melid'].isin(melids_to_keep)]
df = df[~df['melid'].isin(duplicates_by_performer['melid'])]
df = pd.concat([df, filtered_performer_duplicates])
print(f"After handling duplicates by performer and title: {df.shape}")

# Step 3: Handle repeated solos by 'title' but different 'performers' (keep only the oldest performance)
df['releasedate'] = pd.to_datetime(df['releasedate'], errors='coerce')
repeated_titles = df[df.duplicated(subset=['title'], keep=False)]
melids_to_keep_oldest = repeated_titles.loc[
    repeated_titles.groupby('title')['releasedate'].idxmin()
]['melid'].unique()
filtered_oldest_performances = df[df['melid'].isin(melids_to_keep_oldest)]
df = df[~df['melid'].isin(repeated_titles['melid'])]
df = pd.concat([df, filtered_oldest_performances])
print(f"After handling repeated solos by title: {df.shape}")

# Step 4: Remove rows where 'key' is missing or empty
df = df[df['key'].notna() & (df['key'] != '')]
print(f"After removing rows with missing or empty 'key': {df.shape}")

# Step 5: Separate rows with empty 'composer' for later re-integration
no_composer_df = df[df['composer'].isna() | (df['composer'].str.strip() == '')]
df_with_composer = df[~df.index.isin(no_composer_df.index)]
print(f"Rows without composer: {no_composer_df.shape[0]}")
print(f"Rows with composer: {df_with_composer.shape[0]}")

# Step 6: Standardise 'performer' and 'composer' for comparison
def standardise_name(name):
    return (
        str(name).lower()  # Convert to lowercase
        .strip()  # Remove leading/trailing whitespace
        .replace('/', '')  # Remove special characters like '/'
        .replace('.', '')  # Remove periods
    )

df_with_composer['performer_cleaned'] = df_with_composer['performer'].apply(standardise_name)
df_with_composer['composer_cleaned'] = df_with_composer['composer'].apply(standardise_name)

# Step 7: Apply composer match logic
def composer_match(row):
    # Split composer into individual names and clean them
    composer_names = [standardise_name(name) for name in re.split(r'[,/]', row['composer_cleaned'])]
    performer_name = row['performer_cleaned']
    
    # Check if the performer's name matches any of the cleaned composer names
    return any(performer_name in name or name in performer_name for name in composer_names)

df_with_composer['match_status'] = df_with_composer.apply(composer_match, axis=1)
print(f"Number of rows retained after composer match: {df_with_composer['match_status'].sum()}")

df_with_composer = df_with_composer[df_with_composer['match_status']]

# Step 8: Combine rows with matched composer and rows without composer
df_final = pd.concat([df_with_composer, no_composer_df])

# Step 9: Add the 'metrical_position' column for validation
df_final.loc[:, 'metrical_position'] = df_final[['period', 'division', 'bar', 'beat', 'tatum']].astype(str).agg('.'.join, axis=1)
print("Metrical positions have been added to the cleaned dataset.")

# Reset the index for cleanliness
df_final.reset_index(drop=True, inplace=True)

# Drop the temporary standardised columns
df_final.drop(columns=['performer_cleaned', 'composer_cleaned', 'match_status'], inplace=True)

# Save the cleaned dataset to a new CSV file
df_final.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/matched_patterns.py
import pandas as pd
import ast
import os

# Define base and data directories using relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(BASE_DIR, 'data')

# Define file paths
patterns_file = os.path.join(data_dir, 'Patterns', 'patterns_processed.csv')
solos_file = os.path.join(data_dir, 'solos_db_with_durations_cleaned.csv')
output_file = os.path.join(data_dir, 'matched_patterns.csv')

# Load the patterns and solos_db dataframes
patterns_df = pd.read_csv(patterns_file)
solos_db = pd.read_csv(solos_file)

# Normalize the performer and title fields in both dataframes
patterns_df = patterns_df.copy()
patterns_df.loc[:, 'performer'] = patterns_df['performer'].str.lower().str.replace('[^a-z0-9]', '', regex=True).str.strip()
patterns_df.loc[:, 'title'] = patterns_df['title'].str.lower().str.replace('[^a-z]', '', regex=True).str.strip()  # Remove any numbers

solos_db = solos_db.copy()
solos_db.loc[:, 'performer'] = solos_db['performer'].str.lower().str.replace('[^a-z0-9]', '', regex=True).str.strip()
solos_db.loc[:, 'title'] = solos_db['title'].str.lower().str.replace('[^a-z]', '', regex=True).str.strip()  # Remove any numbers

# Initialize an empty list to store matches
matched_data = []

# Iterate over each row in patterns_df
for index, row in patterns_df.iterrows():
    performer = row['performer']
    title = row['title']
    metrical_position = row['metricalposition']
    values = ast.literal_eval(row['value']) if isinstance(row['value'], str) else row['value']
    num_notes = row['N']

    # Find matches in solos_db
    sample_solo = solos_db[(solos_db['performer'] == performer) & (solos_db['title'] == title)]

    if not sample_solo.empty:
        # Start with the first note by finding the matching metrical position row in the solo data
        matched_metrical_row = sample_solo[sample_solo['metrical_position'] == metrical_position]

        if not matched_metrical_row.empty:
            start_index = matched_metrical_row.index[0]  # Get the index of the first note

            # Iterate over the next 'N' notes to get their details
            for i in range(num_notes):
                current_index = start_index + i

                if current_index in sample_solo.index:
                    match = sample_solo.loc[current_index]

                    matched_data.append({
                        'pattern_id': index,
                        'performer': performer,
                        'title': title,
                        'metrical_position': match['metrical_position'],
                        'sixteenth_representation': match['sixteenth_representation'],
                        'melid': match['melid'],
                        'pitch': match['pitch'],
                        'key': match['key'],
                        # Also include from patterns_df: 'N', 'value'
                        'N': row['N'],
                        'value': values  # Keep the original pattern values
                    })

# Convert matched data to DataFrame
matched_full_df = pd.DataFrame(matched_data)

# Save the matched data to a new CSV file
matched_full_df.to_csv(output_file, index=False)

print(f"Matching complete. Results saved to {output_file}")

# File: /Users/paola_amigo/Desktop/Thesis/JazzSolos/src/preprocessing/patterns_processed.py
import pandas as pd

# Load the patterns data from Excel
patterns_df = pd.read_excel('/Users/paola_amigo/Desktop/Thesis/JazzSolos/data/Patterns/pattern_output_pitch_5_10_5_1_db.xlsx', engine='openpyxl')  

# Check the column names to verify if 'id' is present
print(patterns_df.columns)

# Split the 'id' column to get 'performer' and 'title'
if 'id' in patterns_df.columns:
    # Assuming the format is 'PerformerName_SongTitle_FINAL.sv'
    patterns_df[['performer', 'title']] = patterns_df['id'].str.extract(r'([^_]+)_(.*?)_FINAL.sv')

    # Drop the original 'id' column if it's no longer needed
    patterns_df = patterns_df.drop(columns=['id'])

    # Save the cleaned patterns data as a CSV file
    patterns_df.to_csv('/Users/paola_amigo/Desktop/Thesis/JazzSolos/data/patterns/patterns_processed.csv', index=False)
else:
    print("Column 'id' not found. Please check the Excel file.")

