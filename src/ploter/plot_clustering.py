import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict, Counter
from scipy.spatial import distance

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.feature.feature_manager import *
from src.feature.feature_utils import *
from src.feature.clustering import *
from src.ploter.plotting_params import *
from src.ploter.plotting_utils import *
from src.ploter.statistic_annotation import *


def single_plot_embedding(ax: matplotlib.axes.Axes, single_embed: Embedding,
                          cell_types: Dict[CellUID, CellType], cell_type_flag: bool,
                          label_flag: bool = True, s: float = 3, ellipse_flag: bool = True):
    s = s if label_flag else s * 0.75
    embedding_coordinate = single_embed.embedding
    if ellipse_flag:
        for cluster_id in range(single_embed.n_cluster):
            cluster_points = embedding_coordinate[single_embed.labels == cluster_id]
            center = np.mean(cluster_points, axis=0)
            v, w = np.linalg.eigh(np.cov(cluster_points.T))
            v = 2.5 * np.sqrt(v)
            angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
            ellipse = mpatches.Ellipse(center, width=v[0] * 2, height=v[1] * 2, angle=angle,
                                       edgecolor=CLUSTER_COLORLIST[cluster_id],
                                       facecolor=mcolors.to_rgba(CLUSTER_COLORLIST[cluster_id], 0.2),
                                       linestyle='--', linewidth=0.5, alpha=0.2)
            ax.add_patch(ellipse)
    if cell_type_flag:
        celltype_colors = [CELLTYPE2COLOR[cell_types[cell_uid]] for cell_uid in single_embed.cells_uid]
        zorder = [2 if cell_types[cell_uid] is CellType.Calb2_Pos else 0 for cell_uid in single_embed.cells_uid]
        ax.scatter(embedding_coordinate[:, 0], embedding_coordinate[:, 1],
                   s=s, alpha=0.8, edgecolor='none', facecolor=celltype_colors)
    else:
        cluster_colors = [CLUSTER_COLORLIST[single_label] for single_label in single_embed.labels]
        ax.scatter(embedding_coordinate[:, 0], embedding_coordinate[:, 1],
                   s=s, alpha=0.7, edgecolor=cluster_colors, facecolor='none')

    legend_handles = []
    if cell_type_flag:
        n_pos = Counter(cell_types.values()).get(CellType.Calb2_Pos, 0)
        n_neg = Counter(cell_types.values()).get(CellType.Calb2_Neg, 0)
        if n_pos + n_neg > 0:
            legend_handles.append(plt.Line2D(
                [0], [0], linestyle="None", marker="o", markerfacecolor=CELLTYPE2COLOR[CellType.Calb2_Pos],
                markeredgecolor='none', label=f"{CELLTYPE2STR[CellType.Calb2_Pos]} ({n_pos})"))
            legend_handles.append(plt.Line2D(
                [0], [0], linestyle="None", marker="o", markerfacecolor=CELLTYPE2COLOR[CellType.Calb2_Neg],
                markeredgecolor='none', label=f"{CELLTYPE2STR[CellType.Calb2_Neg]} ({n_neg})"))
    else:
        cluster_cells = reverse_dict(single_embed.label_by_cell)
        for cluster_id in sorted(np.unique(single_embed.labels)):
            label_text = f'Noise ({np.sum(single_embed.labels == -1)})' if cluster_id == -1 else \
                f'Cluster {cluster_id+1} ({np.sum(single_embed.labels == cluster_id)})'
            *_, cnt_str = calb2_pos_neg_count(cluster_cells[cluster_id], cell_types, label_flag=False)
            legend_handles.append(plt.Line2D(
                [0], [0], marker='o', linestyle="None",
                label=label_text + cnt_str, markerfacecolor="none", markeredgecolor=CLUSTER_COLORLIST[cluster_id], ))
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    if label_flag:
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(f"{single_embed.score:.3f}", backgroundcolor='gray', color='white')
        ax.legend(handles=legend_handles, frameon=False, fontsize=5 if label_flag else 4, loc='best')


def plot_one_beautiful_embedding(vec_space: VectorSpace, single_embed: Embedding, save_name: str,
                                 size: Tuple[float, float], **kwargs):
    cell_types = vec_space.ref_feature_db.cell_types
    fig, ax = plt.subplots(1, 1)
    if "s" not in kwargs:
        kwargs["s"] = 8
    single_plot_embedding(ax, single_embed, cell_types, **kwargs)

    fig.set_size_inches(*size)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_embedding_summary(
        vec_space: VectorSpace, save_name: str,
        top_k: int = 100,
):
    top_k = min(top_k, vec_space.n_embeddings)
    cell_types = vec_space.ref_feature_db.cell_types
    n_row, n_col = row_col_from_n_subplots(top_k)
    fig, axs = plt.subplots(n_row, n_col)

    for embed_id, single_embedding in enumerate(vec_space.get_embeddings(top_k=top_k)):  # type: int, Embedding
        row_id, col_id = int(np.floor(embed_id/n_col)), embed_id % n_col
        tmp_ax = axs[row_id, col_id]

        single_plot_embedding(tmp_ax, single_embedding, cell_types, cell_type_flag=True, label_flag=True, ellipse_flag=True)

    fig.set_size_inches(3*n_col, 3*n_row)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_umap_space_distance_calb2(
        vec_space: VectorSpace, best_embedding: Embedding, save_name1: str, save_name2: str,
):
    feature_db = vec_space.ref_feature_db

    assert not feature_db.Ai148_flag
    cell_types = feature_db.cell_types
    calb2_p_n = reverse_dict(feature_db.cell_types)
    calb2_pos_cells_uid, calb2_neg_cells_uid = calb2_p_n[CellType.Calb2_Pos], calb2_p_n[CellType.Calb2_Neg]

    clusters_by_cell = reverse_dict(best_embedding.label_by_cell)
    tmp_clusters = [clusters_by_cell[i] for i in range(best_embedding.n_cluster)]

    distance_mean_matrix = np.zeros((2+len(tmp_clusters), 2+len(tmp_clusters)))
    for row_i, group_i in enumerate((calb2_pos_cells_uid, calb2_neg_cells_uid, *tmp_clusters)):
        for col_j, group_j in enumerate((calb2_pos_cells_uid, calb2_neg_cells_uid, *tmp_clusters)):
            distance_mean_matrix[row_i, col_j] = nan_mean(vec_space.pair_distances(group_i, group_j)[:, 0])

    p_value_pos_less_neg = [
        stats.mannwhitneyu(
            vec_space.pair_distances(calb2_pos_cells_uid, single_cluster)[:, 0],
            vec_space.pair_distances(calb2_neg_cells_uid, single_cluster)[:, 0],
            alternative='less', ).pvalue for cluster_id, single_cluster in enumerate(tmp_clusters)
    ]
    p_value_neg_less_pos = [
        stats.mannwhitneyu(
            vec_space.pair_distances(calb2_neg_cells_uid, single_cluster)[:, 0],
            vec_space.pair_distances(calb2_pos_cells_uid, single_cluster)[:, 0],
            alternative='less', ).pvalue for cluster_id, single_cluster in enumerate(tmp_clusters)
    ]
    fig_embed, ax_embed = plt.subplots(1, 1)
    single_plot_embedding(ax_embed, best_embedding, cell_types, cell_type_flag=True, label_flag=False, ellipse_flag=True)
    fig_embed.set_size_inches(1.5, 1.5)
    fig_embed.tight_layout()
    quick_save(fig_embed, save_name1)
    plt.close(fig_embed)

    fig, axs = plt.subplots(2, 3, width_ratios=[4, 0.5, 0.5], height_ratios=[0.5, 4], )
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    ax_heatmap = axs[1, 0]
    ax_marginal_top = axs[0, 0]
    ax_marginal_top.sharex(ax_heatmap)
    ax_marginal_top.tick_params('x', labelbottom=False, length=0)
    axs[0, 1].remove()
    axs[0, 2].remove()
    ax_marginal_right = axs[1, 1]
    ax_marginal_right.sharey(ax_heatmap)
    ax_marginal_right.tick_params('y', labelleft=False, length=0)
    ax_cbar = axs[1, 2]

    # ax_heatmap
    group_labels = ([CELLTYPE2STR[CellType.Calb2_Pos], CELLTYPE2STR[CellType.Calb2_Neg],] +
                    [f"Cluster {i+1}" for i in range(len(tmp_clusters))])
    heatmap_obj = sns.heatmap(distance_mean_matrix, annot=True, fmt=".2f", cmap="viridis_r",
                              xticklabels=group_labels, yticklabels=group_labels,
                              ax=ax_heatmap, cbar=False, square=True)
    ax_heatmap.invert_yaxis()
    plt.setp(ax_heatmap.get_yticklabels(), rotation=0, ha="right")
    plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    # ax_cbar
    cb = fig.colorbar(heatmap_obj.collections[0], cax=ax_cbar, orientation='vertical')
    cb.set_label('Mean Pairwise Distance', fontsize=9)
    ax_cbar.yaxis.set_ticks_position('right')
    ax_cbar.yaxis.set_label_position('right')

    # marginal axs
    for cluster_id in range(len(tmp_clusters)):
        p_text, _, font_kwargs = get_asterisks(p_value_pos_less_neg[cluster_id], simple_flag=True)
        ax_marginal_top.text(cluster_id+2.5, 0, p_text, **font_kwargs,
                             horizontalalignment='center', verticalalignment='center')

        p_text, _, font_kwargs = get_asterisks(p_value_neg_less_pos[cluster_id], simple_flag=True)
        ax_marginal_right.text(0, cluster_id+2.5, p_text, **font_kwargs, rotation=-60,
                               horizontalalignment='center', verticalalignment='center')
    for tmp_ax in (ax_marginal_top, ax_marginal_right):
        tmp_ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax_marginal_top.set_yticks([])
    ax_marginal_right.set_xticks([])

    fig.set_size_inches(4, 3)
    fig.tight_layout()
    quick_save(fig, save_name2)
    plt.close(fig)


def plot_embedding_n_neighbor_distribution(
        vec_space: VectorSpace, save_name: str, top_k: int = 3,
):
    for clustering_method in ("DBSCAN", "KMEANS", "SPECTRAL"):
        average_score = defaultdict(list)
        fig, axs = plt.subplots(top_k, len(PLOTTING_CLUSTERS_OPTIONS),)
        for col_id, cluster_num in enumerate(PLOTTING_CLUSTERS_OPTIONS):
            cell_types = vec_space.ref_feature_db.cell_types
            selected_embeddings = vec_space.get_embeddings(top_k=-1, clustering=clustering_method)
            tmp_embed_list = sorted(general_filter(selected_embeddings, n_cluster=cluster_num),
                                    key=lambda x: x.score, reverse=True)  # type: List[Embedding]
            for row_id, single_embed in enumerate(tmp_embed_list[:top_k]):
                single_plot_embedding(axs[row_id, col_id], single_embed, cell_types,
                                      s=3, cell_type_flag=True, label_flag=False, ellipse_flag=True)

            if clustering_method == "DBSCAN":
                average_score[cluster_num] += [single_embed.score for single_embed in tmp_embed_list]
            elif clustering_method == "KMEANS":
                tmp_list = []
                for single_embed in tmp_embed_list:
                    kmeans = KMeans(n_clusters=cluster_num,
                                    random_state=single_embed.params["umap_random_state"])
                    kmeans.fit(single_embed.embedding)
                    tmp_list.append(kmeans.inertia_)
                average_score[cluster_num] += tmp_list
            elif clustering_method == "SPECTRAL":
                tmp_list = []
                for single_embed in tmp_embed_list:
                    spectral = SpectralClustering(n_clusters=cluster_num,
                                                  random_state=single_embed.params["umap_random_state"])
                    spectral.fit(single_embed.embedding)
                    affinity_matrix = spectral.affinity_matrix_
                    L_norm, dd = laplacian(affinity_matrix, normed=True, return_diag=True)
                    eigenvalues, eigenvectors = eigh(L_norm)
                    tmp_list.append(eigenvalues[cluster_num])
                average_score[cluster_num] += tmp_list

        fig.set_size_inches(4.5, 2.5)
        fig.tight_layout()
        quick_save(fig, save_name+clustering_method+"_top3_example.png")

        fig_s, ax_s = plt.subplots(1, 1,)
        bar_position = np.arange(len(PLOTTING_CLUSTERS_OPTIONS))
        bar_heights = [np.mean(average_score[cluster_num]) for cluster_num in PLOTTING_CLUSTERS_OPTIONS]
        bar_error = [nan_sem(average_score[cluster_num]) for cluster_num in PLOTTING_CLUSTERS_OPTIONS]
        ax_s.bar(bar_position, bar_heights, yerr=bar_error, alpha=0.7, color='black', capsize=2)
        for group_id, cluster_num in enumerate(PLOTTING_CLUSTERS_OPTIONS):
            x_jitter = np.random.normal(loc=group_id, scale=0.15, size=len(average_score[cluster_num]))
            x_jitter = np.clip(x_jitter, group_id - 0.3, group_id + 0.3)

            ax_s.scatter(x_jitter, average_score[cluster_num], alpha=0.1, s=3, color='black', linewidth=0.5)

        bar_dict = {1: average_score[1], **{i: average_score[i] for i in range(len(PLOTTING_CLUSTERS_OPTIONS))}}
        # paired_ttest_with_Bonferroni_correction_simple_version(ax_s, bar_dict)
        ax_s.set_xticks(bar_position, PLOTTING_CLUSTERS_OPTIONS)
        if clustering_method == "DBSCAN":
            ax_s.set_ylabel(f"Silhouette score")
            ax_s.set_xlabel(f"Cluster number")
            ax_s.set_ylim(-0.1, 0.9)
        elif clustering_method == "KMEANS":
            ax_s.set_ylabel(f"Inertia")
            ax_s.set_xlabel(f"Cluster number")
            ax_s.set_ylim(0, 510)
            ax_s.plot(bar_position, bar_heights, lw=1, ls='--', color='red', alpha=0.7, marker='o', markersize=2)
        elif clustering_method == "SPECTRAL":
            ax_s.set_xlabel(f"Eigenvalue Index (Sorted)")
            ax_s.set_ylabel(f"Eigenvalue Magnitude")
            ax_s.plot(bar_position, bar_heights, lw=1, ls='--', color='red', alpha=0.7, marker='o', markersize=2)
        ax_s.yaxis.grid(True, linestyle='--', lw=0.5, color='grey', alpha=0.5)
        ax_s.spines[['right', 'top',]].set_visible(False)
        fig_s.set_size_inches(2.5, 2)
        fig_s.tight_layout()
        quick_save(fig_s, save_name+clustering_method+"_score_dist.png")


def plot_umap_hyperparams_distribution(
        vec_space: VectorSpace, save_name: str,
):
    cell_types = vec_space.ref_feature_db.cell_types
    fig, axs = plt.subplots(len(UMAP_N_NEIGHBORS_OPTIONS), len(UMAP_MIN_DIST_OPTIONS))
    for row_id, n_neighbors_option in enumerate(UMAP_N_NEIGHBORS_OPTIONS):
        for col_id, min_dist_option in enumerate(UMAP_MIN_DIST_OPTIONS):
            single_embed = vec_space.get_embeddings(
                top_k=1, umap_n_neighbors=n_neighbors_option, umap_min_dist=min_dist_option, umap_random_state=42)[0]
            single_plot_embedding(axs[row_id, col_id], single_embed, cell_types,
                                  legend_flag=False, s=3, mini=True)
            # axs[row_id, col_id].xaxis.tick_top()
            # axs[row_id, col_id].xaxis.set_label_position('top')
            # axs[row_id, col_id].set_xlabel(f"UMAP-1")
            axs[row_id, col_id].set_xticks([])
            # axs[row_id, col_id].set_ylabel(f"UMAP-2")
            axs[row_id, col_id].set_yticks([])
            # axs[row_id, col_id].set_title(f"score: {single_embed.score:.2f}")
    fig.set_size_inches(7, 7)
    fig.tight_layout()
    quick_save(fig, save_name)






