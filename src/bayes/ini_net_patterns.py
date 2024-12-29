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