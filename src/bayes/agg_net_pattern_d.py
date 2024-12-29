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