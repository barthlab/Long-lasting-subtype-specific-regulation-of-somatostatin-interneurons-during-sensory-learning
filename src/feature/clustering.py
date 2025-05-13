import numpy as np
from typing import List, Callable, Optional, Dict, Union, Tuple
from dataclasses import dataclass, field, MISSING
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
import os
import os.path as path
from scipy.spatial import distance

from src.config import *
from src.basic.utils import *
from src.data_manager import *
from src.feature.feature_utils import *
from src.feature.feature_manager import *
from src.feature.clustering_params import *
from src.basic.data_operator import *


@dataclass
class Embedding:
    name: str

    n_dim: int
    cells_uid: List[CellUID]
    embedding: np.ndarray
    labels: np.ndarray
    params: dict
    score: float

    def __post_init__(self):
        n_cell = len(self.cells_uid)
        assert self.embedding.shape == (n_cell, self.n_dim)
        assert self.labels.shape == (n_cell,)

    @cached_property
    def n_cluster(self) -> int:
        unique_labels = np.unique(self.labels)
        return len(unique_labels) - (1 if -1 in unique_labels else 0)

    @cached_property
    def embedding_by_cell(self) -> Dict[CellUID, np.ndarray]:
        return {cell_uid: self.embedding[cell_cnt] for cell_cnt, cell_uid in enumerate(self.cells_uid)}

    @cached_property
    def label_by_cell(self) -> Dict[CellUID, float]:
        return {cell_uid: float(self.labels[cell_cnt]) for cell_cnt, cell_uid in enumerate(self.cells_uid)}


def clustering_search(embedding: np.ndarray):
    for cluster_labels, clustering_kwargs in chain(DBSCAN_SEARCH(embedding), KMEANS_SEARCH(embedding),
                                                   SPECTRAL_SEARCH(embedding)):
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        print(f"    Found {n_clusters} clusters at {clustering_kwargs}", end=" -> ")
        if n_clusters < 2:
            print(f"{CLUSTERING_SCORE_NAME} not applicable")
            continue
        score = CLUSTERING_SCORE_FUNC(embedding, cluster_labels)
        print(f"{CLUSTERING_SCORE_NAME}: {score:.4f}")
        yield cluster_labels, score, {**clustering_kwargs, "scoring func": CLUSTERING_SCORE_NAME}


def DBSCAN_SEARCH(embedding: np.ndarray) -> Tuple[np.ndarray, dict]:
    for eps_option in DBSCAN_EPS_OPTIONS:
        cluster_method = DBSCAN(eps=eps_option)
        cluster_labels = cluster_method.fit_predict(embedding)
        yield cluster_labels, {"clustering": "DBSCAN", "cluster-kw: dbscan_eps": eps_option}


def KMEANS_SEARCH(embedding: np.ndarray) -> Tuple[np.ndarray, dict]:
    for n_clusters_option in KMEANS_N_CLUSTERS_OPTIONS:
        cluster_method = KMeans(n_clusters=n_clusters_option, random_state=CLUSTERING_RANDOM_SEED)
        cluster_labels = cluster_method.fit_predict(embedding)
        yield cluster_labels, {"clustering": "KMEANS", "cluster-kw: kmeans_n_clusters": n_clusters_option,
                               "cluster-kw: kmeans_random_state": CLUSTERING_RANDOM_SEED}


def SPECTRAL_SEARCH(embedding: np.ndarray):
    for n_clusters_option in SPECTRAL_N_CLUSTERS_OPTIONS:
        cluster_method = SpectralClustering(n_clusters=n_clusters_option, random_state=CLUSTERING_RANDOM_SEED)
        cluster_labels = cluster_method.fit_predict(embedding)
        yield cluster_labels, {"clustering": "SPECTRAL", "cluster-kw: spectral_n_clusters": n_clusters_option,
                               "cluster-kw: spectral_random_state": CLUSTERING_RANDOM_SEED}


# def GM_SEARCH(embedding: np.ndarray):
#     for n_components_option in GAUSSIAN_MIXTURE_N_COMPONENTS_OPTIONS:
#         cluster_method = GaussianMixture(n_components=n_components_option, random_state=CLUSTERING_RANDOM_SEED)
#         cluster_labels = cluster_method.fit_predict(embedding)
#         yield cluster_labels, {"clustering": "GaussianMixture", "cluster-kw: gm_n_clusters": n_components_option,
#                                "cluster-kw: gm_random_state": CLUSTERING_RANDOM_SEED}


@dataclass
class VectorSpace:
    name: str

    vector_data: Dict[CellUID, np.ndarray]
    ref_feature_db: FeatureDataBase

    ndim: int = field(init=False)
    _embeddings: List[Embedding] = field(init=False, repr=False)

    def __post_init__(self):
        assert set(self.ref_feature_db.cells_uid) == set(self.vector_data.keys())
        self.ndim = len(self.vector_data[self.ref_feature_db.cells_uid[0]])
        for cell_uid, single_vector in self.vector_data.items():
            assert single_vector.shape == (self.ndim,)
        self._embeddings = []

    @property
    def n_embeddings(self) -> int:
        return len(self._embeddings)

    def get_embeddings(self, top_k: int = 1, **criteria) -> List[Embedding]:
        top_k = top_k if top_k > 0 else self.n_embeddings
        satisfied_embeddings = []
        for single_embed in self._embeddings:
            if all(single_embed.params.get(k, None) == v for k, v in criteria.items()):
                satisfied_embeddings.append(single_embed)
            if len(satisfied_embeddings) >= top_k:
                return satisfied_embeddings
        return satisfied_embeddings

    @property
    def cells_uid(self) -> List[CellUID]:
        return self.ref_feature_db.cells_uid

    @property
    def by_cells(self) -> Dict[CellUID, np.ndarray]:
        return self.vector_data

    @cached_property
    def matrix(self) -> np.ndarray:
        vector_list = [self.vector_data[cell_uid] for cell_uid in self.ref_feature_db.cells_uid]
        return np.stack(vector_list, axis=0)

    @cached_property
    def grid_search_path(self) -> str:
        return path.join(FEATURE_EXTRACTED_PATH, self.ref_feature_db.ref_img.exp_id,
                         f"{self.ref_feature_db.name}_{self.name}_embeddings.xlsx")

    def archive_exists(self) -> bool:
        return path.exists(self.grid_search_path)

    def prepare_embedding(self, overwrite=False):
        if (not overwrite) and (self.archive_exists()):
            self._load_exist_embeddings()
        else:
            self._grid_search_embeddings()
            self._save_embeddings()

    def _grid_search_embeddings(self):
        self._embeddings = []
        for n_neighbor_option in UMAP_N_NEIGHBORS_OPTIONS:
            for min_dist_option in UMAP_MIN_DIST_OPTIONS:
                for umap_random_seed in UMAP_RANDOM_SEED_OPTIONS:
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbor_option,
                        min_dist=min_dist_option,
                        n_components=2,
                        random_state=umap_random_seed,
                    )
                    embedding = reducer.fit_transform(self.matrix)
                    print(f"Searching at UMAP(nn={n_neighbor_option}, md={min_dist_option}, seed={umap_random_seed})")
                    for cluster_labels, score, clustering_kwargs in clustering_search(embedding):
                        new_embedding = Embedding(
                            name=f"embedding {len(self._embeddings)}",
                            n_dim=2,
                            cells_uid=self.cells_uid,
                            embedding=np.copy(embedding),
                            labels=np.array(cluster_labels),
                            params={
                                'umap_n_neighbors': n_neighbor_option,
                                'umap_min_dist': min_dist_option,
                                'umap_random_state': umap_random_seed,
                                **clustering_kwargs
                            },
                            score=score
                        )
                        self._embeddings.append(new_embedding)
        self._embeddings = sorted(self._embeddings, key=lambda x: x.score, reverse=True)

    def _save_embeddings(self):
        assert len(self._embeddings) > 0
        print(f"Saving {len(self._embeddings)} embeddings into {self.grid_search_path}")
        os.makedirs(path.dirname(self.grid_search_path), exist_ok=True)
        padding_empty_list = ["", ]*(len(self.cells_uid)-1)
        with pd.ExcelWriter(self.grid_search_path, engine='xlsxwriter') as writer:
            for single_embedding in self._embeddings:
                prefix_dict = {
                    "name": [single_embedding.name, ] + padding_empty_list,
                    "n_dim": [single_embedding.n_dim, ] + padding_empty_list,
                    "score": [single_embedding.score, ] + padding_empty_list,
                }
                params_dict = {
                    param_key: [param_value, ] + padding_empty_list for param_key, param_value in
                    single_embedding.params.items()
                }
                embedding_dict = {
                    f"embedding-{i}": single_embedding.embedding[:, i]
                    for i in range(single_embedding.n_dim)
                }
                label_dict = {
                    f"labels": single_embedding.labels
                }
                df = pd.DataFrame({**prefix_dict, **params_dict,
                                   **decompose_cell_uid_list(single_embedding.cells_uid),
                                   **embedding_dict, **label_dict})
                df.to_excel(writer, sheet_name=single_embedding.name, index=False)
        print(f"Embedding sheets saved at: {self.grid_search_path}.")

    def _load_exist_embeddings(self):
        print(f"Loading embeddings from {self.grid_search_path}...")
        xls = pd.ExcelFile(self.grid_search_path, engine='openpyxl')
        self._embeddings = []
        for sheet_id, sheet_name in enumerate(xls.sheet_names):
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                raise ValueError
            embedding_name = df["name"][0]
            tmp_n_dim = int(df["n_dim"][0])
            tmp_score = df["score"][0]
            tmp_params_dict = {
                'umap_n_neighbors': int(df["umap_n_neighbors"][0]),
                'umap_min_dist': df["umap_min_dist"][0],
                'umap_random_state': int(df["umap_random_state"][0]),
                'clustering': df["clustering"][0],
                'scoring func': df["scoring func"][0],
            }
            for col_name, col_value in df.items():
                if "cluster-kw: " in col_name:
                    tmp_params_dict[col_name] = col_value[0]
            # check cell uid
            tmp_cells_uid = synthesize_cell_uid_list({_k: df[_k].tolist()
                                                      for _k in ("exp_id", "mice_id", "fov_id", "cell_id")})
            assert tmp_cells_uid == self.cells_uid, \
                f"Different cells_uid\nStored File: {tmp_cells_uid}\nCurrent Feature Database: {self.cells_uid}"

            tmp_embedding = np.stack([df[f"embedding-{i}"] for i in range(tmp_n_dim)], axis=1)
            new_embedding = Embedding(
                name=embedding_name,
                n_dim=tmp_n_dim,
                cells_uid=tmp_cells_uid,
                embedding=tmp_embedding,
                labels=df["labels"],
                params=tmp_params_dict,
                score=tmp_score
            )
            self._embeddings.append(new_embedding)

            if sheet_id % 100 == 0:
                print(f"{sheet_id+1}/{len(xls.sheet_names)} Loading embedding: {embedding_name} -> "
                      f"{CLUSTERING_SCORE_NAME}: {new_embedding.score:.4f}")
        print("Loading complete.")

    def pair_distances(self, cells_uid_1: List[CellUID], cells_uid_2: List[CellUID] | None = None, ):
        _self_flag = (cells_uid_2 is None) or (cells_uid_2 == cells_uid_1)
        distance_pairs = []
        for cell_uid_1 in cells_uid_1[:-1]:
            if _self_flag:
                cells_uid_2 = cells_uid_1[cells_uid_1.index(cell_uid_1) + 1:]
            for cell_uid_2 in cells_uid_2:
                tmp_euclidean_dist = distance.euclidean(
                    self.by_cells[cell_uid_1],
                    self.by_cells[cell_uid_2]
                )
                tmp_chebyshev_dist = distance.chebyshev(
                    self.by_cells[cell_uid_1],
                    self.by_cells[cell_uid_2]
                )
                distance_pairs.append((tmp_euclidean_dist, tmp_chebyshev_dist))
        return np.stack(distance_pairs, axis=0)


def get_feature_vector(
        feature_db: FeatureDataBase, selected_feature_names: List[str],
        selected_days: str, vector_name: str) -> VectorSpace:
    all_feature_by_cells = combine_dicts(*[
        feature_db.get(single_feature, day_postfix=selected_days).standardized().by_cells
        for single_feature in selected_feature_names
    ])
    feature_vector = {}
    for cell_uid, feature_list in all_feature_by_cells.items():  # type: CellUID, list
        assert isinstance(cell_uid, CellUID)
        assert len(feature_list) == len(selected_feature_names)
        assert np.sum(np.isnan(feature_list)) == 0
        feature_vector[cell_uid] = np.array(feature_list)
    return VectorSpace(name=vector_name, vector_data=feature_vector, ref_feature_db=feature_db)


def get_waveform_vector(feature_db: FeatureDataBase, selected_days: str, vector_name: str) -> VectorSpace:
    all_waveform_by_cells = extract_avg_df_f0(
        feature_db.ref_img, days_dict={selected_days: feature_db.days_ref[selected_days]},
        trial_type=EventType.Puff)[selected_days]
    _, _, (xs, interp_matrix) = sync_timeseries(
        [single_ts for cell_uid, single_ts in all_waveform_by_cells.items()])
    waveform_vector = {}
    for cell_idx, (cell_uid, single_ts) in enumerate(all_waveform_by_cells.items()):
        waveform_vector[cell_uid] = interp_matrix[cell_idx]
    return VectorSpace(name=vector_name, vector_data=waveform_vector, ref_feature_db=feature_db)



