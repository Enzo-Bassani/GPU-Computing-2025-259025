import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import warnings
from random import shuffle

# Use seaborn style
sns.set(style="whitegrid")
# sns.set(style="ticks")
# colors = ["#e63946",  # vibrant red
#           "#f4a261",  # warm orange
#           "#2a9d8f",  # teal-green
#           "#264653",  # dark blue-gray
#           "#e9c46a",  # golden yellow
#           "#1d3557",  # deep navy
#           "#ff6b6b",  # coral red
#           "#43aa8b",  # soft green
#           "#f3722c",  # vivid orange
#           "#577590"]  # steel blue
# colors = [
#     "#e63946",  # vivid red
#     "#2a9d8f",  # teal green
#     "#457b9d",  # strong blue
#     "#f4a261",  # warm orange
#     "#1d3557",  # deep navy
#     "#4ecdc4",  # aqua
#     "#a8dadc",  # light cyan
#     "#f3722c",  # vivid orange
#     "#43aa8b",  # rich green
#     "#277da1"   # strong cool blue
# ]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

shuffle(colors)

def extract_matrix_order(matrix_str):
    """Extract the number prefix used for ordering matrices."""
    match = re.search(r'^(\d+)_', matrix_str)
    if match:
        return int(match.group(1))
    return float('inf')  # Place matrices without a number prefix at the end

def clean_matrix_name(matrix_str):
    """Extract and clean the matrix name from the full string."""
    # Extract just the name without dimensions
    match = re.search(r'(\d+)_([A-Za-z0-9_]+)', matrix_str)
    if match:
        return match.group(2).replace('_', ' ')
    return matrix_str

def load_json_files(directory):
    """Load all JSON files from the specified directory."""
    data: list[dict] = []
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

def group_by_matrix(stats_data: list[dict]):
    """Group the statistics by matrix."""
    matrices: dict[str, list[dict]] = defaultdict(list)
    for stat in stats_data:
        if 'matrix' in stat:
            matrix_name = stat['matrix']
            matrices[matrix_name].append(stat)
        else:
            warnings.warn(f"Skipping entry without matrix field: {stat}")
    return matrices

def plot_and_save_bar(programs, data, title, ylabel, filename, color=None, log_scale=False, cache_misses=None, cache_misses_lld=None):
    """Helper function to plot and save bar charts with optional cache miss overlay."""
    # Validate cache_misses input
    if cache_misses is not None and len(cache_misses) != len(programs):
        print(f"Warning: cache_misses length ({len(cache_misses)}) doesn't match programs length ({len(programs)}). Ignoring cache_misses.")
        cache_misses = None
        cache_misses_lld = None  # If one is invalid, ignore both since they should always be present together
    # Validate cache_misses_lld input
    elif cache_misses_lld is not None and len(cache_misses_lld) != len(programs):
        print(f"Warning: cache_misses_lld length ({len(cache_misses_lld)}) doesn't match programs length ({len(programs)}). Ignoring cache_misses_lld.")
        cache_misses_lld = None
        cache_misses = None  # If one is invalid, ignore both since they should always be present together
    plt.figure(figsize=(12, 7))

    # Use a color palette if color not specified
    if color is None:
        bars = plt.bar(programs, data, edgecolor='black', linewidth=1, zorder=3)  # Higher zorder to appear above grid
        # Set colors from palette
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])
    else:
        bars = plt.bar(programs, data, color=color, edgecolor='black', linewidth=1, zorder=3)  # Higher zorder to appear above grid

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel("Algorithm", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    # Grid removed as it's unnecessary with the labels

    # For log scale, we need to handle zero values specially
    if log_scale:
        # Check if there are any non-zero values to plot
        non_zero_data = [d for d in data if d > 0]
        if non_zero_data:
            plt.yscale('log')
            plt.ylabel(f"{ylabel} (log scale)", fontsize=14)
            # If we have zero values, add a note to the title
            if len(non_zero_data) < len(data):
                plt.title(f"{title}\n(Zero values excluded from log scale)", fontsize=16, fontweight='bold')
        else:
            # If all values are zero, don't use log scale
            log_scale = False

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        if log_scale:
            label_pos = yval * 1.1
        else:
            label_pos = yval + max(data)*0.01
        # Format with 5 decimal places
        plt.text(bar.get_x() + bar.get_width()/2.0, label_pos,
                 f'{yval:.5f}', ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=1000, clip_on=False)  # Higher zorder for labels

    # Add cache_miss data if available (both cache_miss and cache_miss_lld should be present together)
    if cache_misses and any(x is not None for x in cache_misses) and cache_misses_lld and any(x is not None for x in cache_misses_lld):
        try:
            # Create a secondary y-axis for cache misses
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            # Get positions of bars
            x_positions = [bar.get_x() + bar.get_width()/2.0 for bar in bars]

            # Filter out None values for plotting cache_miss
            valid_positions = []
            valid_misses = []
            program_labels = []
            for x, miss, prog in zip(x_positions, cache_misses, programs):
                if miss is not None:
                    try:
                        miss_float = float(miss)  # Ensure it's a number
                        valid_positions.append(x)
                        miss_pct = miss_float
                        valid_misses.append(miss_pct)
                        program_labels.append(prog)
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid cache miss value '{miss}' for program '{prog}', skipping")

            # Filter out None values for plotting cache_miss_lld
            valid_positions_lld = []
            valid_misses_lld = []
            program_labels_lld = []
            for x, miss, prog in zip(x_positions, cache_misses_lld, programs):
                if miss is not None:
                    try:
                        miss_float = float(miss)  # Ensure it's a number
                        valid_positions_lld.append(x)
                        miss_pct = miss_float
                        valid_misses_lld.append(miss_pct)
                        program_labels_lld.append(prog)
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid cache miss LLD value '{miss}' for program '{prog}', skipping")

            if valid_positions:
                # Plot cache misses as a line with dots
                line = ax2.plot(valid_positions, valid_misses, 'ro-', linewidth=2, markersize=6, label='Cache Misses %', zorder=4)  # Higher zorder for the line

                # Set label for secondary y-axis
                ax2.set_ylabel('Cache Misses (%)', fontsize=14)
                ax2.tick_params(axis='y')

                # Add percentage labels ABOVE the points for cache_miss
                for x, y, prog in zip(valid_positions, valid_misses, program_labels):
                    ax2.annotate(f'{y:.2f}%',
                                (x, y),
                                xytext=(0, 25),  # Greater positive vertical offset to place above the point
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                color='r',
                                fontweight='bold',
                                fontsize=9,
                                zorder=5)  # Higher zorder for annotations

            # Plot cache_miss_lld as a green line with dots (if we have data)
            if valid_positions_lld:
                line_lld = ax2.plot(valid_positions_lld, valid_misses_lld, 'go-', linewidth=2,
                                markersize=6, label='Cache Misses LLD %', zorder=4)

                # Add percentage labels ABOVE the points for cache_miss_lld
                for x, y, prog in zip(valid_positions_lld, valid_misses_lld, program_labels_lld):
                    ax2.annotate(f'{y:.2f}%',
                                (x, y),
                                xytext=(0, 10),  # Smaller positive vertical offset since green line is usually below red
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                color='g',
                                fontweight='bold',
                                fontsize=9,
                                zorder=5)

                # Update legend to include both lines
                ax2.legend(loc='upper left')

        except Exception as e:
            print(f"Error plotting cache misses: {str(e)}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()

def create_plots_for_matrix(matrix_name, stats, output_dir="plots"):
    """Create and save plots for a specific matrix."""
    # Use ticks style (no grid) for individual matrix plots
    original_style = sns.axes_style()
    sns.set_style("ticks")
    # Extract data
    programs = [stat['algorithm'] for stat in stats]
    exec_times = [stat['time_ms'] for stat in stats]
    gflops = [stat.get('gflops', 0) for stat in stats]  # Use get with default for missing keys
    bandwidths = [stat.get('bandwidth_gbps', 0) for stat in stats]
    cache_misses = [stat.get('cache_miss', None) for stat in stats]  # Extract cache_miss if available
    cache_misses_lld = [stat.get('cache_miss_lld', None) for stat in stats]  # Extract cache_miss_lld if available

    # Sort all data by execution time (ascending)
    sorted_indices = np.argsort(exec_times)
    sorted_programs = [programs[i] for i in sorted_indices]
    sorted_exec_times = [exec_times[i] for i in sorted_indices]
    sorted_gflops = [gflops[i] for i in sorted_indices]
    sorted_bandwidths = [bandwidths[i] for i in sorted_indices]
    sorted_cache_misses = [cache_misses[i] for i in sorted_indices]
    sorted_cache_misses_lld = [cache_misses_lld[i] for i in sorted_indices]

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
        log_scale=True,
        cache_misses=sorted_cache_misses,
        cache_misses_lld=sorted_cache_misses_lld
    )

    if any(gflops):  # Only create plot if there's actual data
        plot_and_save_bar(
            sorted_programs,
            sorted_gflops,
            f"GFLOP/s for {display_name}",
            "GFLOP/s",
            f"{matrix_dir}/gflops.png",
            "#2ca02c",  # green
            cache_misses=sorted_cache_misses,
            cache_misses_lld=sorted_cache_misses_lld
        )

    if any(bandwidths):  # Only create plot if there's actual data
        plot_and_save_bar(
            sorted_programs,
            sorted_bandwidths,
            f"Bandwidth (Gbps) for {display_name}",
            "Bandwidth (Gbps)",
            f"{matrix_dir}/bandwidth.png",
            "#ff7f0e",  # orange
            cache_misses=sorted_cache_misses,
            cache_misses_lld=sorted_cache_misses_lld
        )

    # Reset to original style
    sns.set_style(original_style)
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
                'original_matrix': matrix_name,  # Keep original name for sorting
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

        # Sort by the numeric prefix in the original matrix name
        matrix_stats.sort(key=lambda x: extract_matrix_order(x['original_matrix']))

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
    for matrix_name, stats in sorted(matrices.items(), key=lambda x: extract_matrix_order(x[0])):
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
        # For log scale, we can only plot positive values
        if plt.gca().get_yscale() == 'log':
            # For each value, only plot it if it's greater than 0
            for idx, v in enumerate(values):
                if v > 0:
                    pos = x_positions[idx]
                    plt.bar(pos, v, width=width, label=algo if idx == 0 else "",
                            color=algo_colors[algo], edgecolor='black', linewidth=0.5)
        else:
            # Normal scale - plot all values
            plt.bar(x_positions, values, width=width, label=algo,
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
        # For log scale, we can only plot positive values
        if plt.gca().get_yscale() == 'log':
            # For each value, only plot it if it's greater than 0
            for idx, v in enumerate(values):
                if v > 0:
                    pos = x_positions[idx]
                    plt.bar(pos, v, width=width, label=algo if idx == 0 else "",
                            color=algo_colors[algo], edgecolor='black', linewidth=0.5)
        else:
            # Normal scale - plot all values
            plt.bar(x_positions, values, width=width, label=algo,
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
        # For log scale, we can only plot positive values
        if plt.gca().get_yscale() == 'log':
            # For each value, only plot it if it's greater than 0
            for idx, v in enumerate(values):
                if v > 0:
                    pos = x_positions[idx]
                    plt.bar(pos, v, width=width, label=algo if idx == 0 else "",
                            color=algo_colors[algo], edgecolor='black', linewidth=0.5)
        else:
            # Normal scale - plot all values
            plt.bar(x_positions, values, width=width, label=algo,
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

def filter_out_exceptions(matrices:dict[str, list[dict]], matrix_exceptions: list[str], algo_exceptions: list[str]):
    for exception in matrix_exceptions:
        del matrices[exception]
    for exception in algo_exceptions:
        for key in matrices.keys():
            matrices[key] = list(filter(lambda d: d.get("algorithm") != exception, matrices[key]))

MATRIX_EXCEPTIONS = [
    "1_1138_bus",
    # "6_923136_Emilia_923"
]

ALGO_EXCEPTIONS = [
    # "cpu_naive",
    # "cpu_naive_columns",
    # "cpu_coo",
    # "cpu_csr",
    # "cpu_coo_struct"
]

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

        filter_out_exceptions(matrices, MATRIX_EXCEPTIONS, ALGO_EXCEPTIONS)

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
        # print("Creating summary comparison plots...")
        # create_summary_plots(matrices, output_dir)

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
