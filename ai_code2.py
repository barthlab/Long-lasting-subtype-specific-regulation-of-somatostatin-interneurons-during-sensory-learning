import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import mannwhitneyu
from itertools import combinations

# --- 1. Simulate Data ---
# Imagine we have 100 cells and 50 features (e.g., genes)
n_cells = 100
n_features = 50

# Generate random data (replace this with your actual data)
# Rows = cells, Columns = features
data = np.random.rand(n_cells, n_features) * 10

# Assign cell types (replace with your actual labels)
# Let's say the first 30 cells are 'type1', the rest are 'other'
labels = np.array(['type1'] * 30 + ['other'] * 70)

# Get indices for each group
type1_indices = np.where(labels == 'type1')[0]
other_indices = np.where(labels == 'other')[0]

# --- 2. Calculate Metrics for Pairs ---

# Group 1: type1 vs type1 pairs
type1_type1_distances = []
type1_type1_similarities = []

# Use combinations to get all unique pairs within type1
for i, j in combinations(type1_indices, 2):
    cell_i = data[i, :]
    cell_j = data[j, :]

    dist = euclidean(cell_i, cell_j)
    type1_type1_distances.append(dist)

    # Cosine similarity is 1 - cosine distance
    sim = 1 - cosine(cell_i, cell_j)
    type1_type1_similarities.append(sim)

# Group 2: type1 vs other pairs
type1_other_distances = []
type1_other_similarities = []

for i in type1_indices:
    for j in other_indices:
        cell_i = data[i, :]
        cell_j = data[j, :]

        dist = euclidean(cell_i, cell_j)
        type1_other_distances.append(dist)

        # Cosine similarity is 1 - cosine distance
        sim = 1 - cosine(cell_i, cell_j)
        type1_other_similarities.append(sim)

# --- 3. Perform Statistical Tests ---

print(f"Number of type1 vs type1 pairs: {len(type1_type1_distances)}")
print(f"Number of type1 vs other pairs: {len(type1_other_distances)}\n")

# Check if we have enough data in both groups to perform the test
if not type1_type1_distances or not type1_other_distances:
    print("Not enough pairs in one or both groups to perform statistical tests.")
else:
    # -- Euclidean Distance Test --
    # Hypothesis: Median distance is different between groups
    # We expect type1_type1 distances might be smaller if type1 cells cluster
    stat_dist, p_dist = mannwhitneyu(type1_type1_distances,
                                     type1_other_distances,
                                     alternative='less')  # Test if group1 < group2

    print("--- Euclidean Distance Comparison ---")
    print(f"Average Type1-Type1 Distance: {np.mean(type1_type1_distances):.4f}")
    print(f"Average Type1-Other Distance: {np.mean(type1_other_distances):.4f}")
    print(f"Mann-Whitney U test (Type1-Type1 < Type1-Other):")
    print(f"  Statistic: {stat_dist:.4f}")
    print(f"  P-value: {p_dist:.4g}")  # Use general format for p-value

    if p_dist < 0.05:
        print("  Result: Significantly smaller distances within Type1 pairs (p < 0.05).")
    else:
        print("  Result: No significant difference in distances (p >= 0.05).")
    print("-" * 35)

    # -- Cosine Similarity Test --
    # Hypothesis: Median similarity is different between groups
    # We expect type1_type1 similarities might be higher if type1 cells have similar patterns
    stat_sim, p_sim = mannwhitneyu(type1_type1_similarities,
                                   type1_other_similarities,
                                   alternative='greater')  # Test if group1 > group2

    print("\n--- Cosine Similarity Comparison ---")
    print(f"Average Type1-Type1 Similarity: {np.mean(type1_type1_similarities):.4f}")
    print(f"Average Type1-Other Similarity: {np.mean(type1_other_similarities):.4f}")
    print(f"Mann-Whitney U test (Type1-Type1 > Type1-Other):")
    print(f"  Statistic: {stat_sim:.4f}")
    print(f"  P-value: {p_sim:.4g}")  # Use general format for p-value

    if p_sim < 0.05:
        print("  Result: Significantly greater similarity within Type1 pairs (p < 0.05).")
    else:
        print("  Result: No significant difference in similarities (p >= 0.05).")
    print("-" * 35)