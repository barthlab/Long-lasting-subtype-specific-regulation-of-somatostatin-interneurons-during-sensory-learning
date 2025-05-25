from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# UMAP related
UMAP_N_NEIGHBORS_OPTIONS = (6, 7, 8, 9, 10, 11)
UMAP_MIN_DIST_OPTIONS = (0.001, 0.005, 0.01, 0.05, 0.1,)
UMAP_RANDOM_SEED_OPTIONS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


# clustering params related
DBSCAN_EPS_OPTIONS = [0.1, 0.5, 1, 2, 4,]
KMEANS_N_CLUSTERS_OPTIONS = [2, 3, 4, 5, 6, ]
SPECTRAL_N_CLUSTERS_OPTIONS = [2, 3, 4, 5, 6, ]
# GAUSSIAN_MIXTURE_N_COMPONENTS_OPTIONS = [2, 3, 4, 5, 6, ]
CLUSTERING_RANDOM_SEED = 42
PLOTTING_CLUSTERS_OPTIONS = [2, 3, 4, 5, 6, ]

# score related
CLUSTERING_SCORE_FUNC = silhouette_score
CLUSTERING_SCORE_NAME = CLUSTERING_SCORE_FUNC.__name__

# prediction related
BEST_NUM_CLUSTERS = 3
TOP_LABELLING_OCCURRENCE_THRESHOLD = 2
FEATURE_SELECTED_DAYS = "ACC456"
