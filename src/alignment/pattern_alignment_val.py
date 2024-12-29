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