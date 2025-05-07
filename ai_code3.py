import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh # For eigenvalues

# Generate sample data
# Using make_blobs to create data that is somewhat clearly clustered for demonstration
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.70, random_state=42)

# --- 1. Inertia (Elbow Method) ---
print("--- 1. Inertia (Elbow Method) ---")
inertia_values = []
k_range = range(1, 11) # Test k from 1 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_values, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.show()
print("Interpretation: Look for the 'elbow' point where the rate of decrease in inertia slows down.")
print("In this example, k=4 seems like a good elbow point.\n")

# --- 2. Silhouette Score ---
print("--- 2. Silhouette Score ---")
silhouette_scores = []
# We need at least 2 clusters for silhouette score
k_range_silhouette = range(2, 11)

for k in k_range_silhouette:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_range_silhouette, silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.xticks(k_range_silhouette)
plt.grid(True)
plt.show()
print("Interpretation: The k that maximizes the average silhouette score is often chosen.")
print("In this example, k=4 gives the highest silhouette score.\n")

# --- 3. Eigenvalue Gap (using a simplified approach with Spectral Clustering's affinity matrix) ---
# Note: Full eigenvalue gap analysis is more involved with graph Laplacians.
# Here, we'll look at the eigenvalues of the affinity matrix as a proxy,
# though true spectral analysis often looks at Laplacian eigenvalues.
# For SpectralClustering, n_clusters is a parameter, so we don't directly find it FROM eigenvalues here,
# but we can illustrate what eigenvalues might look like.
# A more rigorous approach would be to compute the Laplacian and its eigenvalues separately.

print("--- 3. Eigenvalue Gap (Illustrative) ---")
# For illustration, let's compute an affinity matrix and its eigenvalues.
# This is more complex to directly tie to a simple k selection without more spectral graph theory.
# scikit-learn's SpectralClustering does this internally.

# We'll use the affinity matrix from SpectralClustering with a default number of clusters
# and then examine its eigenvalues as an example. This is not the typical way to *select* k
# using eigengap with sklearn's SpectralClustering directly, but to show what eigenvalues are.

# A more direct way to show an eigengap for choosing k:
n_components_to_check = 10 # Check up to this many eigenvalues
affinity_matrix = SpectralClustering(n_clusters=4, # A placeholder, as we are interested in eigenvalues
                                   affinity='rbf', # Radial Basis Function kernel
                                   random_state=42,
                                   n_components=n_components_to_check, # Ask for more components
                                   assign_labels='kmeans').fit(X).affinity_matrix_

# eigenvalues of the affinity matrix (related to, but not exactly the Laplacian eigenvalues used in theory)
# For a proper eigengap heuristic, one typically uses the eigenvalues of the Graph Laplacian.
# Let's compute the unnormalized graph Laplacian: D - A
# D is the degree matrix, A is the affinity matrix.
D = np.diag(np.sum(affinity_matrix, axis=1))
L = D - affinity_matrix

# Get eigenvalues, sorted
eigenvalues, eigenvectors = eigh(L) # eigh for symmetric matrices, sorts eigenvalues
sorted_eigenvalues = np.sort(eigenvalues) # eigh already sorts them for symmetric

# We usually look for a gap after the k-th smallest non-zero eigenvalue.
# The number of zero eigenvalues often indicates the number of connected components.
# For practical k selection, we look for a large gap: lambda_k+1 - lambda_k

# Let's plot the first few eigenvalues (often the smallest ones are most informative for Laplacians)
n_eigenvalues_to_plot = min(n_components_to_check, len(sorted_eigenvalues))

plt.figure(figsize=(8, 5))
plt.plot(range(1, n_eigenvalues_to_plot + 1), sorted_eigenvalues[:n_eigenvalues_to_plot], marker='o', linestyle='--')
plt.xlabel('Eigenvalue Index (Sorted)')
plt.ylabel('Eigenvalue Magnitude')
plt.title('Sorted Eigenvalues of Graph Laplacian (Illustrative Eigengap)')
plt.xticks(range(1, n_eigenvalues_to_plot + 1))
plt.grid(True)
plt.show()

# Calculate gaps
gaps = np.diff(sorted_eigenvalues[:n_eigenvalues_to_plot])
print("Eigenvalues:", sorted_eigenvalues[:n_eigenvalues_to_plot])
print("Gaps between consecutive eigenvalues:", gaps)
# The "k" would be chosen where the gap is largest.
# For example, if the 4th eigenvalue is small and the 5th is much larger, this suggests k=4 clusters.
# The first eigenvalue of the Laplacian is often 0 (or close to 0 due to numerical precision)
# for a connected graph. The number of zero eigenvalues indicates the number of connected components.
# So, we look for the gap after the k-th *smallest non-zero* eigenvalue, often k corresponding to the
# index BEFORE the largest jump if we want k clusters.
# For example, if eigenvalues are [0.01, 0.02, 0.03, 0.8, 1.2], the gap after the 3rd (0.03) is large.
# This suggests k=3 (or if 0.01 implies one component, then k=3 components).
# If the first eigenvalue is ~0, and the next k-1 are small, and then there's a jump, it suggests k clusters.
# For our data generated with 4 centers, we'd hope to see a significant gap after the 4th eigenvalue
# (or 3rd if indexing from 0 for the non-zero ones).
# The index of the largest gap suggests the number of clusters.
if len(gaps) > 0:
    optimal_k_eigengap = np.argmax(gaps) + 1 # +1 because diff reduces length by 1
    # If the first few eigenvalues are very close to zero, indicating multiple connected components
    # (which is what spectral clustering looks for), then 'optimal_k_eigengap' can be interpreted
    # as the number of clusters.
    print(f"Based on the largest gap in the first {n_eigenvalues_to_plot} eigenvalues, an optimal k might be: {optimal_k_eigengap}")
else:
    print("Not enough eigenvalues to compute gaps.")

print("Interpretation: Look for the largest jump (gap) between sorted eigenvalues.")
print("The number of eigenvalues before this largest gap often suggests the optimal number of clusters.")
print("For our data generated with 4 centers, we'd hope to see a significant gap making k=4 evident.")
print("Note: The first eigenvalue of the graph Laplacian is usually 0 for a connected graph.\n")


# --- 4. BIC (Bayesian Information Criterion) ---
# Typically used with model-based clustering like Gaussian Mixture Models (GMM)
print("--- 4. BIC (Bayesian Information Criterion) ---")
bic_scores = []
k_range_gmm = range(1, 11) # Test k from 1 to 10

for k in k_range_gmm:
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full') # 'full' is common
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    print(f"For n_components = {k}, BIC = {gmm.bic(X):.2f}")


plt.figure(figsize=(8, 5))
plt.plot(k_range_gmm, bic_scores, marker='o', linestyle='--')
plt.xlabel('Number of Components (Clusters k)')
plt.ylabel('BIC Score')
plt.title('BIC for Optimal Number of GMM Components')
plt.xticks(k_range_gmm)
plt.grid(True)
plt.show()
print("Interpretation: The k that minimizes the BIC score is generally preferred.")
print("In this example, k=4 yields the lowest BIC score.\n")

print("--- Summary of Optimal k from different methods for the generated data ---")
# Optimal k from Elbow (visual inspection) usually around 4
# Optimal k from Silhouette Score (highest value) usually 4
# Optimal k from Eigengap (visual inspection or largest gap index) ideally 4
# Optimal k from BIC (lowest value) usually 4
print("For this synthetic dataset with 4 true centers, most methods should ideally point to k=4.")