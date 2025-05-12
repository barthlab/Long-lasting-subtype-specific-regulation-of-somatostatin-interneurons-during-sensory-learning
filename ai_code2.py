import matplotlib.pyplot as plt
import numpy as np
import random

# Sample Data: A list of lists, where each inner list contains data points for a bar.
data = [
    [10, 12, 11, 13, 12, 14, 11.5],
    [15, 17, 16, 18, 17.5, 19, 16.5, 17],
    [8, 9, 7, 8.5, 9.5, 7.5],
    [12, 13, 11, 12.5, 13.5, 10.5, 11.5, 12.8]
]

# Bar labels (optional)
labels = ['Group A', 'Group B', 'Group C', 'Group D']

# --- Plotting ---

# Create a figure and an axes
fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size for better look

# Bar Plot
# Calculate the mean of each group for the bar heights
bar_heights = [np.mean(group) for group in data]
bar_positions = np.arange(len(data))

# Choose some nice colors
bar_colors = ['#8ECDDD', '#FFB3BA', '#90EE90', '#FFDFBA'] # Pastel colors
scatter_colors = ['#2E86C1', '#E74C3C', '#27AE60', '#E67E22'] # Darker, matching colors

bars = ax.bar(bar_positions, bar_heights, color=bar_colors, edgecolor='black', alpha=0.7, capsize=5)

# Scatter/Swarm Plot Overlay
for i, group_data in enumerate(data):
    # Add some horizontal jitter to x-coordinates for better visualization (swarm-like)
    # The amount of jitter can be adjusted
    jitter_strength = 0.15
    x_jitter = np.random.normal(loc=i, scale=jitter_strength, size=len(group_data))
    # Ensure jittered points stay visually associated with their bar
    x_jitter = np.clip(x_jitter, i - jitter_strength * 2, i + jitter_strength * 2)

    ax.scatter(x_jitter, group_data, color=scatter_colors[i], zorder=2, alpha=0.8, s=50, edgecolor='black', linewidth=0.5)

# --- Styling and Labels ---

# Set title and labels
ax.set_title('Bar Graph with Scatter Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('Groups', fontsize=14)
ax.set_ylabel('Values', fontsize=14)

# Set x-axis tick labels
ax.set_xticks(bar_positions)
ax.set_xticklabels(labels, fontsize=12)

# Add a grid for better readability (optional)
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
ax.set_axisbelow(True) # Send grid lines behind bars

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a legend (optional, if needed to distinguish scatter points further, though colors help)
# For this simple case, a legend might be overkill, but here's how you could add one if you had distinct series:
# ax.legend()

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Show the plot
plt.show()
