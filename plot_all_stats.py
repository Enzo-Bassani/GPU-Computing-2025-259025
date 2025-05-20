import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import warnings

# Use seaborn style
sns.set(style="whitegrid")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def clean_matrix_name(matrix_str):
    """Extract and clean the matrix name from the full string."""
    # Extract just the name without dimensions
    match = re.search(r'(\d+)_([A-Za-z0-9_]+)', matrix_str)
    if match:
        return match.group(2).replace('_', ' ')
    return matrix_str

def load_json_files(directory):
    """Load all JSON files from the specified directory."""
    data = []
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(directory, filename), 'r') as f:
                    data.append(json.load(f))
            except json.JSONDecodeError:
                warnings.warn(f"Failed to parse JSON from {filename}, skipping")
            except Exception as e:
                warnings.warn(f"Error reading {filename}: {str(e)}")
    
    if not data:
        raise ValueError(f"No valid JSON files found in {directory}")
    
    return data

def group_by_matrix(stats_data):
    """Group the statistics by matrix."""
    matrices = defaultdict(list)
    for stat in stats_data:
        if 'matrix' in stat:
            matrix_name = stat['matrix']
            matrices[matrix_name].append(stat)
        else:
            warnings.warn(f"Skipping entry without matrix field: {stat}")
    return matrices

def plot_and_save_bar(programs, data, title, ylabel, filename, color=None, log_scale=False):
    """Helper function to plot and save bar charts."""
    plt.figure(figsize=(12, 7))
    
    # Use a color palette if color not specified
    if color is None:
        bars = plt.bar(programs, data, edgecolor='black', linewidth=1)
        # Set colors from palette
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])
    else:
        bars = plt.bar(programs, data, color=color, edgecolor='black', linewidth=1)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel("Algorithm", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if log_scale and min(data) > 0:
        plt.yscale('log')
        plt.ylabel(f"{ylabel} (log scale)", fontsize=14)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        if log_scale:
            label_pos = yval * 1.1
        else:
            label_pos = yval + max(data)*0.01
        plt.text(bar.get_x() + bar.get_width()/2.0, label_pos, 
                 f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()

def create_plots_for_matrix(matrix_name, stats, output_dir="plots"):
    """Create and save plots for a specific matrix."""
    # Extract data
    programs = [stat['algorithm'] for stat in stats]
    exec_times = [stat['time_ms'] for stat in stats]
    gflops = [stat.get('gflops', 0) for stat in stats]  # Use get with default for missing keys
    bandwidths = [stat.get('bandwidth_gbps', 0) for stat in stats]
    
    # Sort all data by execution time (ascending)
    sorted_indices = np.argsort(exec_times)
    sorted_programs = [programs[i] for i in sorted_indices]
    sorted_exec_times = [exec_times[i] for i in sorted_indices]
    sorted_gflops = [gflops[i] for i in sorted_indices]
    sorted_bandwidths = [bandwidths[i] for i in sorted_indices]
    
    # Clean matrix name for display
    display_name = clean_matrix_name(matrix_name)
    
    # Create output directory for this matrix
    matrix_dir = os.path.join(output_dir, matrix_name)
    os.makedirs(matrix_dir, exist_ok=True)
    
    # Create and save plots
    plot_and_save_bar(
        sorted_programs,
        sorted_exec_times,
        f"Execution Time (ms) for {display_name}",
        "Time (ms)",
        f"{matrix_dir}/execution_time.png",
        "#1f77b4",  # blue
        log_scale=True
    )
    
    if any(gflops):  # Only create plot if there's actual data
        plot_and_save_bar(
            sorted_programs,
            sorted_gflops,
            f"GFLOP/s for {display_name}",
            "GFLOP/s",
            f"{matrix_dir}/gflops.png",
            "#2ca02c"  # green
        )
    
    if any(bandwidths):  # Only create plot if there's actual data
        plot_and_save_bar(
            sorted_programs,
            sorted_bandwidths,
            f"Bandwidth (Gbps) for {display_name}",
            "Bandwidth (Gbps)",
            f"{matrix_dir}/bandwidth.png",
            "#ff7f0e"  # orange
        )
    
    return display_name

def create_summary_plots(matrices, output_dir="plots"):
    """Create summary plots comparing the same algorithm across different matrices."""
    # Group by algorithm instead of matrix
    algorithms = defaultdict(list)
    
    for matrix_name, stats in matrices.items():
        display_name = clean_matrix_name(matrix_name)
        for stat in stats:
            algo = stat['algorithm']
            algorithms[algo].append({
                'matrix': display_name,
                'time_ms': stat['time_ms'],
                'gflops': stat.get('gflops', 0),
                'bandwidth_gbps': stat.get('bandwidth_gbps', 0)
            })
    
    # Create summary directory
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # For each algorithm that has enough data points, create comparison plots
    for algo, matrix_stats in algorithms.items():
        if len(matrix_stats) < 2:  # Skip if only one matrix
            continue
            
        # Sort by matrix name for consistency
        matrix_stats.sort(key=lambda x: x['matrix'])
        
        matrix_names = [stat['matrix'] for stat in matrix_stats]
        exec_times = [stat['time_ms'] for stat in matrix_stats]
        gflops = [stat['gflops'] for stat in matrix_stats]
        bandwidths = [stat['bandwidth_gbps'] for stat in matrix_stats]
        
        # Create plots
        plot_and_save_bar(
            matrix_names,
            exec_times,
            f"Execution Time (ms) for {algo} across Matrices",
            "Time (ms)",
            f"{summary_dir}/{algo}_execution_time.png",
            log_scale=True
        )
        
        if any(gflops):
            plot_and_save_bar(
                matrix_names,
                gflops,
                f"GFLOP/s for {algo} across Matrices",
                "GFLOP/s",
                f"{summary_dir}/{algo}_gflops.png"
            )
        
        if any(bandwidths):
            plot_and_save_bar(
                matrix_names,
                bandwidths,
                f"Bandwidth (Gbps) for {algo} across Matrices",
                "Bandwidth (Gbps)",
                f"{summary_dir}/{algo}_bandwidth.png"
            )

def create_comprehensive_plots(matrices, output_dir="plots"):
    """Create comprehensive plots showing all algorithms across all matrices."""
    # First collect all unique algorithms
    all_algorithms = set()
    for matrix_stats in matrices.values():
        for stat in matrix_stats:
            all_algorithms.add(stat['algorithm'])
    
    all_algorithms = sorted(list(all_algorithms))
    
    # Create a mapping for algorithm to color
    algo_colors = {}
    for i, algo in enumerate(all_algorithms):
        algo_colors[algo] = colors[i % len(colors)]
    
    # Prepare data for grouped bar charts
    matrix_names = []
    grouped_bandwidth_data = defaultdict(list)
    grouped_time_data = defaultdict(list)
    grouped_gflops_data = defaultdict(list)
    
    # For each matrix, collect data for each algorithm
    for matrix_name, stats in sorted(matrices.items()):
        display_name = clean_matrix_name(matrix_name)
        matrix_names.append(display_name)
        
        # Create a mapping of algorithm to its stats for this matrix
        algo_to_stats = {stat['algorithm']: stat for stat in stats}
        
        # For each algorithm, get the data if available, otherwise use 0
        for algo in all_algorithms:
            if algo in algo_to_stats:
                grouped_bandwidth_data[algo].append(algo_to_stats[algo].get('bandwidth_gbps', 0))
                grouped_time_data[algo].append(algo_to_stats[algo].get('time_ms', 0))
                grouped_gflops_data[algo].append(algo_to_stats[algo].get('gflops', 0))
            else:
                grouped_bandwidth_data[algo].append(0)  # No data for this algorithm on this matrix
                grouped_time_data[algo].append(0)
                grouped_gflops_data[algo].append(0)
    
    # Calculate the width of each bar and positions
    n_algorithms = len(all_algorithms)
    width = 0.8 / n_algorithms
    
    # 1. Create the BANDWIDTH grouped bar chart
    plt.figure(figsize=(15, 10))
    
    # For each algorithm, create a group of bars
    for i, algo in enumerate(all_algorithms):
        x_positions = np.arange(len(matrix_names)) + (i - n_algorithms/2 + 0.5) * width
        plt.bar(x_positions, grouped_bandwidth_data[algo], width=width, label=algo, 
                color=algo_colors[algo], edgecolor='black', linewidth=0.5)
    
    # Set chart properties
    plt.xlabel('Matrix', fontsize=14)
    plt.ylabel('Bandwidth (Gbps)', fontsize=14)
    plt.title('Bandwidth Comparison Across All Matrices and Algorithms', fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(matrix_names)), matrix_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add legend with algorithm names
    plt.legend(title='Algorithms', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Ensure everything fits
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/comprehensive_bandwidth_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a log scale version for better visualization of differences
    plt.figure(figsize=(15, 10))
    
    # For each algorithm, create a group of bars
    for i, algo in enumerate(all_algorithms):
        x_positions = np.arange(len(matrix_names)) + (i - n_algorithms/2 + 0.5) * width
        values = grouped_bandwidth_data[algo]
        # Replace zeros with small value for log scale
        log_values = [v if v > 0 else 0.01 for v in values]
        plt.bar(x_positions, log_values, width=width, label=algo, 
                color=algo_colors[algo], edgecolor='black', linewidth=0.5)
    
    # Set chart properties
    plt.xlabel('Matrix', fontsize=14)
    plt.ylabel('Bandwidth (Gbps) - Log Scale', fontsize=14)
    plt.title('Bandwidth Comparison Across All Matrices and Algorithms (Log Scale)', 
              fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(matrix_names)), matrix_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    
    # Add legend with algorithm names
    plt.legend(title='Algorithms', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Ensure everything fits
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/comprehensive_bandwidth_comparison_log.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create the EXECUTION TIME grouped bar chart
    plt.figure(figsize=(15, 10))
    
    # For each algorithm, create a group of bars
    for i, algo in enumerate(all_algorithms):
        x_positions = np.arange(len(matrix_names)) + (i - n_algorithms/2 + 0.5) * width
        plt.bar(x_positions, grouped_time_data[algo], width=width, label=algo, 
                color=algo_colors[algo], edgecolor='black', linewidth=0.5)
    
    # Set chart properties
    plt.xlabel('Matrix', fontsize=14)
    plt.ylabel('Execution Time (ms)', fontsize=14)
    plt.title('Execution Time Comparison Across All Matrices and Algorithms', fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(matrix_names)), matrix_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add legend with algorithm names
    plt.legend(title='Algorithms', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Ensure everything fits
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/comprehensive_execution_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a log scale version for better visualization of differences
    plt.figure(figsize=(15, 10))
    
    # For each algorithm, create a group of bars
    for i, algo in enumerate(all_algorithms):
        x_positions = np.arange(len(matrix_names)) + (i - n_algorithms/2 + 0.5) * width
        values = grouped_time_data[algo]
        # Replace zeros with small value for log scale
        log_values = [v if v > 0 else 0.01 for v in values]
        plt.bar(x_positions, log_values, width=width, label=algo, 
                color=algo_colors[algo], edgecolor='black', linewidth=0.5)
    
    # Set chart properties
    plt.xlabel('Matrix', fontsize=14)
    plt.ylabel('Execution Time (ms) - Log Scale', fontsize=14)
    plt.title('Execution Time Comparison Across All Matrices and Algorithms (Log Scale)', 
              fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(matrix_names)), matrix_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    
    # Add legend with algorithm names
    plt.legend(title='Algorithms', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Ensure everything fits
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/comprehensive_execution_time_comparison_log.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create the GFLOPS grouped bar chart
    plt.figure(figsize=(15, 10))
    
    # For each algorithm, create a group of bars
    for i, algo in enumerate(all_algorithms):
        x_positions = np.arange(len(matrix_names)) + (i - n_algorithms/2 + 0.5) * width
        plt.bar(x_positions, grouped_gflops_data[algo], width=width, label=algo, 
                color=algo_colors[algo], edgecolor='black', linewidth=0.5)
    
    # Set chart properties
    plt.xlabel('Matrix', fontsize=14)
    plt.ylabel('GFLOP/s', fontsize=14)
    plt.title('GFLOP/s Comparison Across All Matrices and Algorithms', fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(matrix_names)), matrix_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add legend with algorithm names
    plt.legend(title='Algorithms', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Ensure everything fits
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/comprehensive_gflops_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a log scale version for better visualization of differences
    plt.figure(figsize=(15, 10))
    
    # For each algorithm, create a group of bars
    for i, algo in enumerate(all_algorithms):
        x_positions = np.arange(len(matrix_names)) + (i - n_algorithms/2 + 0.5) * width
        values = grouped_gflops_data[algo]
        # Replace zeros with small value for log scale
        log_values = [v if v > 0 else 0.01 for v in values]
        plt.bar(x_positions, log_values, width=width, label=algo, 
                color=algo_colors[algo], edgecolor='black', linewidth=0.5)
    
    # Set chart properties
    plt.xlabel('Matrix', fontsize=14)
    plt.ylabel('GFLOP/s - Log Scale', fontsize=14)
    plt.title('GFLOP/s Comparison Across All Matrices and Algorithms (Log Scale)', 
              fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(matrix_names)), matrix_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    
    # Add legend with algorithm names
    plt.legend(title='Algorithms', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Ensure everything fits
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/comprehensive_gflops_comparison_log.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Define parameters
    stats_dir = "./stats"
    output_dir = "./plots"
    
    try:
        # Load all JSON files from stats directory
        print(f"Loading JSON files from {stats_dir}...")
        stats_data = load_json_files(stats_dir)
        
        # Group by matrix
        matrices = group_by_matrix(stats_data)
        
        print(f"Found {len(matrices)} different matrices")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots for each matrix
        for matrix_name, stats in matrices.items():
            if len(stats) < 2:
                print(f"Skipping matrix {matrix_name} with insufficient data points")
                continue
                
            print(f"Creating plots for matrix: {matrix_name}")
            create_plots_for_matrix(matrix_name, stats, output_dir)
        
        # Create summary comparison plots
        print("Creating summary comparison plots...")
        create_summary_plots(matrices, output_dir)
        
        # Create comprehensive comparison plots
        print("Creating comprehensive comparison plots...")
        create_comprehensive_plots(matrices, output_dir)
        
        print(f"All plots have been saved in the '{output_dir}' directory")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()