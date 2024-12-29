import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

# Define relative paths
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