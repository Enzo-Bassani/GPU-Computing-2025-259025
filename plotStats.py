import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style
sns.set(style="whitegrid")

# Benchmark data
programs = [
    "gpu_csr",
    "gpu_csr_stride",
    "gpu_coo_add_atomic",
    "cpu_csr",
    "cpu_coo"
]

execution_times = [
    0.350616,
    0.421476,
    1.592214,
    16.232254,
    30.621483
]

flop_per_s = [
    119.584882,
    99.479819,
    26.333359,
    2.583026,
    1.369246
]

bandwidths = [
    749.104088,
    623.161871,
    263.333586,
    16.180605,
    13.692460
]

# Helper function to plot and save bar charts
def plot_and_save_bar(data, title, ylabel, filename, color):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(programs, data, color=color, edgecolor='black', linewidth=1.2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add value labels on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(data)*0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Create and save enhanced plots
plot_and_save_bar(execution_times,
                  "Execution Time (geo_avg in ms) for Programs on Matrix Emilia",
                  "Time (ms)",
                  "execution_time_emilia.png",
                  "#1f77b4")  # blue

plot_and_save_bar(bandwidths,
                  "Bandwidth (Gbps) for Programs on Matrix Emilia",
                  "Bandwidth (Gbps)",
                  "bandwidth_emilia.png",
                  "#ff7f0e")  # orange

plot_and_save_bar(flop_per_s,
                  "FLOP/s for Programs on Matrix Emilia",
                  "FLOP/s (in billions)",
                  "flops_emilia.png",
                  "#2ca02c")  # green

print("Beautiful plots saved as: execution_time_emilia.png, bandwidth_emilia.png, flops_emilia.png")
