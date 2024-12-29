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