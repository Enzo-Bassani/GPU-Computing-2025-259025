import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style
sns.set(style="whitegrid")

# Updated programs and their stats for matrix bcsstk17
programs = [
    "gpu_csr",
    "gpu_csr_stride",
    "gpu_csr_constant_memory",
    "gpu_coo_add_atomic",
    "gpu_naive",
    "cpu_csr",
    "cpu_coo",
    "cpu_naive"
]

execution_times = [
    0.016685,
    0.019556,
    0.022830,
    0.026210,
    1.285733,
    0.147159,
    0.313557,
    136.973083
]

flop_per_s = [
    26.348916,
    22.480203,
    19.256049,
    16.773413,
    187.330772,
    2.987399,
    1.402053,
    1.758428
]

bandwidths = [
    165.986232,
    141.615093,
    121.304385,
    167.734125,
    749.357227,
    18.819261,
    14.020529,
    7.034034
]

# Helper function to plot and save bar charts
def plot_and_save_bar(data, title, ylabel, filename, color):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(programs, data, color=color, edgecolor='black', linewidth=1.2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(data)*0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Create and save enhanced plots
plot_and_save_bar(execution_times,
                  "Execution Time (geo_avg in ms) for Programs on Matrix bcsstk17",
                  "Time (ms)",
                  "execution_time_bcsstk17.png",
                  "#1f77b4")  # blue

plot_and_save_bar(bandwidths,
                  "Bandwidth (Gbps) for Programs on Matrix bcsstk17",
                  "Bandwidth (Gbps)",
                  "bandwidth_bcsstk17.png",
                  "#ff7f0e")  # orange

plot_and_save_bar(flop_per_s,
                  "FLOP/s for Programs on Matrix bcsstk17",
                  "FLOP/s (in billions)",
                  "flops_bcsstk17.png",
                  "#2ca02c")  # green

print("Beautiful plots saved as: execution_time_bcsstk17.png, bandwidth_bcsstk17.png, flops_bcsstk17.png")
