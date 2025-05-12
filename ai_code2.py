import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import mannwhitneyu

# --- 0. Define Group Labels ---
group_labels = ['Positive', 'Negative', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
num_groups = len(group_labels)
positive_idx = 0
negative_idx = 1
# cluster_group_indices are the actual indices in the group_labels list for C1-C4
cluster_group_indices = [2, 3, 4, 5]
cluster_names_list = [group_labels[i] for i in cluster_group_indices]

# --- 1. Generate Sample Data (Using the same function as before) ---
def generate_sample_distances(num_groups, num_pairs_per_group=50):
    np.random.seed(42) # Keep seed for reproducibility
    all_distances = [[[] for _ in range(num_groups)] for _ in range(num_groups)]
    # Adjusted means to potentially show clearer effects for the new tests
    mean_dist_map = np.array([
      # Pos  Neg   C1   C2   C3   C4
        [0.5, 2.0, 0.8, 1.5, 1.8, 1.0], # Positive to ... (Pos-C1=0.8, Pos-Neg=2.0)
        [2.0, 0.6, 1.6, 0.9, 1.2, 1.9], # Negative to ... (Neg-C2=0.9, Neg-Pos=2.0)
        [0.8, 1.6, 0.7, 1.0, 1.1, 0.9], # Cluster 1
        [1.5, 0.9, 1.0, 0.6, 0.8, 1.4], # Cluster 2
        [1.8, 1.2, 1.1, 0.8, 0.7, 1.3], # Cluster 3
        [1.0, 1.9, 0.9, 1.4, 1.3, 0.5]  # Cluster 4
    ])
    for i in range(num_groups):
        for j in range(i, num_groups): # Iterate only upper triangle
            mean_val = mean_dist_map[i,j]
            std_val = mean_val * 0.3 # Arbitrary std dev
            distances = np.random.normal(loc=mean_val, scale=std_val, size=num_pairs_per_group)
            distances = np.clip(distances, 0.01, None) # Ensure positive distances
            all_distances[i][j] = list(distances)
            if i != j:
                all_distances[j][i] = list(distances) # Symmetric
    return all_distances

cell_pair_distances = generate_sample_distances(num_groups)

# --- 2. Calculate Mean Distances for Heatmap ---
mean_distance_matrix = np.zeros((num_groups, num_groups))
for i in range(num_groups):
    for j in range(num_groups):
        if cell_pair_distances[i][j]:
            mean_distance_matrix[i, j] = np.mean(cell_pair_distances[i][j])
        else:
            mean_distance_matrix[i, j] = np.nan

# --- 3. Perform Statistical Tests ---
epsilon = np.finfo(float).eps # Small number to avoid log10(0)

# 3a. Tests for Top Marginal Plot: D(Pos, ClusterK) < D(Pos, Negative)
p_values_pos_to_clusters = {}
neg_log10_p_pos_to_clusters = {}
dist_pos_to_neg_ref = np.array(cell_pair_distances[positive_idx][negative_idx])

for k_idx in cluster_group_indices:
    cluster_name = group_labels[k_idx]
    dist_pos_to_k = np.array(cell_pair_distances[positive_idx][k_idx])
    p_val = 1.0 # Default if test can't be run
    if len(dist_pos_to_k) > 1 and len(dist_pos_to_neg_ref) > 1 and \
       len(np.unique(dist_pos_to_k)) > 1 and len(np.unique(dist_pos_to_neg_ref)) > 1:
        _, p_val = mannwhitneyu(dist_pos_to_k, dist_pos_to_neg_ref, alternative='less', nan_policy='omit')
    p_values_pos_to_clusters[cluster_name] = p_val
    neg_log10_p_pos_to_clusters[cluster_name] = -np.log10(p_val + epsilon)

# 3b. Tests for Right Marginal Plot: D(Neg, ClusterK) < D(Neg, Positive)
p_values_neg_to_clusters = {}
neg_log10_p_neg_to_clusters = {}
dist_neg_to_pos_ref = np.array(cell_pair_distances[negative_idx][positive_idx])

for k_idx in cluster_group_indices:
    cluster_name = group_labels[k_idx]
    dist_neg_to_k = np.array(cell_pair_distances[negative_idx][k_idx])
    p_val = 1.0 # Default if test can't be run
    if len(dist_neg_to_k) > 1 and len(dist_neg_to_pos_ref) > 1 and \
       len(np.unique(dist_neg_to_k)) > 1 and len(np.unique(dist_neg_to_pos_ref)) > 1:
        _, p_val = mannwhitneyu(dist_neg_to_k, dist_neg_to_pos_ref, alternative='less', nan_policy='omit')
    p_values_neg_to_clusters[cluster_name] = p_val
    neg_log10_p_neg_to_clusters[cluster_name] = -np.log10(p_val + epsilon)


# --- 4. Create the Plot ---
fig = plt.figure(figsize=(9, 9))
# GridSpec: 2 rows, 2 columns. Top-left for top marginal, bottom-left for heatmap,
# bottom-right for right marginal, top-right for colorbar.
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       wspace=0.05, hspace=0.05) # Minimal space between plots

ax_top = plt.subplot(gs[0, 0])
ax_heatmap = plt.subplot(gs[1, 0])
ax_right = plt.subplot(gs[1, 1])
ax_cbar = plt.subplot(gs[0, 1]) # Slot for the colorbar

# 4a. Heatmap
heatmap_obj = sns.heatmap(mean_distance_matrix, annot=True, fmt=".2f", cmap="viridis_r",
                          xticklabels=False,
                          yticklabels=group_labels,
                          ax=ax_heatmap,
                          cbar=False) # We'll draw colorbar in ax_cbar
plt.setp(ax_heatmap.get_yticklabels(), rotation=0, ha='right')

# Add colorbar to the dedicated ax_cbar slot
cb = fig.colorbar(heatmap_obj.collections[0], cax=ax_cbar, orientation='vertical')
cb.set_label('Mean Pairwise Distance', fontsize=9)
ax_cbar.yaxis.set_ticks_position('left') # Move cbar ticks and label to left side
ax_cbar.yaxis.set_label_position('left')
ax_cbar.tick_params(labelsize=8)


# 4b. Top Marginal Plot
# X-coordinates for points align with cluster columns in heatmap
x_coords_top = np.array(cluster_group_indices)
y_values_top = [neg_log10_p_pos_to_clusters[name] for name in cluster_names_list]
raw_p_top = [p_values_pos_to_clusters[name] for name in cluster_names_list]

ax_top.plot(x_coords_top, y_values_top, marker='o', linestyle='None', color='crimson', markersize=6, clip_on=False)
for i, x_coord in enumerate(x_coords_top):
    ax_top.text(x_coord, y_values_top[i] - 0.3, f"p={raw_p_top[i]:.1e}", ha='center', va='top', fontsize=7, color='crimson')

sig_line_val = -np.log10(0.05 + epsilon)
ax_top.axhline(sig_line_val, color='grey', linestyle='--', linewidth=0.8)
# Place p=0.05 text annotation
ax_top.text(num_groups - 0.5, sig_line_val + 0.1, 'p=0.05', color='grey', ha='right', va='bottom', fontsize=7)

ax_top.set_ylabel('Pos Affinity\n-log10(P)', fontsize=8)
ax_top.set_xticks(np.arange(num_groups))
ax_top.set_xticklabels(group_labels, rotation=45, ha='left') # ha='left' for top rotated labels
ax_top.xaxis.tick_top()
ax_top.xaxis.set_label_position('top')
ax_top.set_xlim(ax_heatmap.get_xlim()) # Align x-limits with heatmap
max_y_val_top = max(sig_line_val + 0.5, (max(y_values_top) + 1) if y_values_top else (sig_line_val + 0.5))
ax_top.set_ylim(0, max_y_val_top)
ax_top.spines[['bottom', 'right']].set_visible(False) # Cleaner look


# 4c. Right Marginal Plot
# Y-coordinates for points align with cluster rows in heatmap
y_coords_right = np.array(cluster_group_indices)
x_values_right = [neg_log10_p_neg_to_clusters[name] for name in cluster_names_list]
raw_p_right = [p_values_neg_to_clusters[name] for name in cluster_names_list]

ax_right.plot(x_values_right, y_coords_right, marker='o', linestyle='None', color='dodgerblue', markersize=6, clip_on=False)
for i, y_coord in enumerate(y_coords_right):
    ax_right.text(x_values_right[i] + 0.2, y_coord, f"p={raw_p_right[i]:.1e}", ha='left', va='center', fontsize=7, color='dodgerblue')

ax_right.axvline(sig_line_val, color='grey', linestyle='--', linewidth=0.8)
# Place p=0.05 text annotation
ax_right.text(sig_line_val + 0.05, num_groups - 0.5, 'p=0.05', color='grey', ha='left', va='top', rotation=90, fontsize=7)

ax_right.set_xlabel('Neg Affinity\n-log10(P)', fontsize=8)
ax_right.set_yticks(np.arange(num_groups))
ax_right.set_yticklabels(group_labels, rotation=0, ha='left') # ha='left' moves text away from axis line
ax_right.yaxis.tick_right()
ax_right.yaxis.set_label_position('right')
ax_right.set_ylim(ax_heatmap.get_ylim()) # Align y-limits with heatmap
max_x_val_right = max(sig_line_val + 0.5, (max(x_values_right) + 1) if x_values_right else (sig_line_val + 0.5))
ax_right.set_xlim(0, max_x_val_right)
ax_right.spines[['left', 'top']].set_visible(False) # Cleaner look

# Overall Title and Layout
fig.suptitle('Cell Distance Matrix & Cluster Affinities', fontsize=14, y=1.02) # Adjusted y for suptitle
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
plt.show()

# Print p-values for reference
print("Top Marginal: P-values for H_a: Dist(Positive, ClusterK) < Dist(Positive, Negative)")
for name in cluster_names_list:
    print(f"  {name}: {p_values_pos_to_clusters[name]:.4f}")

print("\nRight Marginal: P-values for H_a: Dist(Negative, ClusterK) < Dist(Negative, Positive)")
for name in cluster_names_list:
    print(f"  {name}: {p_values_neg_to_clusters[name]:.4f}")