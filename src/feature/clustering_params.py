from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# UMAP related
UMAP_N_NEIGHBORS_OPTIONS = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
UMAP_MIN_DIST_OPTIONS = (0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2,)
UMAP_RANDOM_SEED_OPTIONS = (42, 1234, 0,)

# clustering params related
DBSCAN_EPS_OPTIONS = [0.5, 1, 2, 4, 8]
KMEANS_N_CLUSTERS_OPTIONS = [2, 3, 4, 5, 6, ]
SPECTRAL_N_CLUSTERS_OPTIONS = [2, 3, 4, 5, 6, ]
# GAUSSIAN_MIXTURE_N_COMPONENTS_OPTIONS = [2, 3, 4, 5, 6, ]
CLUSTERING_RANDOM_SEED = 42

# score related
CLUSTERING_SCORE_FUNC = silhouette_score
CLUSTERING_SCORE_NAME = CLUSTERING_SCORE_FUNC.__name__

