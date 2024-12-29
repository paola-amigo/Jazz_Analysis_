import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt

# Define relative paths
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