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


def single_plot_embedding(ax: matplotlib.axes.Axes, single_embed: Embedding, cell_types: Dict[CellUID, CellType]):
    embedding_coordinate = single_embed.embedding
    celltype_colors = [CELLTYPE2COLOR[cell_types[cell_uid]] for cell_uid in single_embed.cells_uid]

    for cluster_id in range(single_embed.n_cluster):
        cluster_points = embedding_coordinate[single_embed.labels == cluster_id]
        center = np.mean(cluster_points, axis=0)
        v, w = np.linalg.eigh(np.cov(cluster_points.T))
        v = 2.5 * np.sqrt(v)
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
        ellipse = mpatches.Ellipse(center, width=v[0] * 2, height=v[1] * 2, angle=angle,
                                   edgecolor=CLUSTER_COLORLIST[cluster_id],
                                   facecolor=mcolors.to_rgba(CLUSTER_COLORLIST[cluster_id], 0.2),
                                   linestyle='--', linewidth=2, alpha=0.2)
        ax.add_patch(ellipse)

    ax.scatter(embedding_coordinate[:, 0], embedding_coordinate[:, 1],
               s=15, alpha=0.7, edgecolor='none', facecolor=celltype_colors)

    legend_handles = []
    cluster_cells = reverse_dict(single_embed.label_by_cell)
    for cluster_id in sorted(np.unique(single_embed.labels)):
        label_text = f'Noise ({np.sum(single_embed.labels == -1)})' if cluster_id == -1 else \
            f'Cluster {cluster_id+1} ({np.sum(single_embed.labels == cluster_id)})'
        *_, cnt_str = calb2_pos_neg_count(cluster_cells[cluster_id], cell_types)
        legend_handles.append(plt.Line2D(
            [0], [0], marker='o', linestyle="None", alpha=0.2,
            label=label_text + cnt_str, markeredgecolor="none", markerfacecolor=CLUSTER_COLORLIST[cluster_id], ))

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"Score: {single_embed.score:.3f} ({single_embed.n_cluster} clusters)")
    ax.set_aspect("equal", adjustable='box')
    ax.spines[['right', 'top']].set_visible(False)

    n_pos = Counter(cell_types.values())[CellType.Calb2_Pos]
    n_neg = Counter(cell_types.values())[CellType.Calb2_Neg]
    legend_handles.append(plt.Line2D(
        [0], [0], linestyle="None", marker="o", markerfacecolor=CELLTYPE2COLOR[CellType.Calb2_Pos],
        markeredgecolor='none', label=f"{CELLTYPE2STR[CellType.Calb2_Pos]} ({n_pos})"))
    legend_handles.append(plt.Line2D(
        [0], [0], linestyle="None", marker="o", markerfacecolor=CELLTYPE2COLOR[CellType.Calb2_Neg],
        markeredgecolor='none', label=f"{CELLTYPE2STR[CellType.Calb2_Neg]} ({n_neg})"))
    ax.legend(handles=legend_handles, frameon=False, fontsize=5, title_fontsize=5, loc='best')


def plot_embedding_summary(
        vec_space: VectorSpace, save_name: str,
        top_k: int = 30,
):
    top_k = min(top_k, vec_space.n_embeddings)
    cell_types = vec_space.ref_feature_db.cell_types
    n_row, n_col = row_col_from_n_subplots(top_k)
    fig, axs = plt.subplots(n_row, n_col)

    for embed_id, single_embedding in enumerate(vec_space.get_embeddings(top_k=top_k)):  # type: int, Embedding
        row_id, col_id = int(np.floor(embed_id/n_col)), embed_id % n_col
        tmp_ax = axs[row_id, col_id]

        single_plot_embedding(tmp_ax, single_embedding, cell_types)

    fig.set_size_inches(3*n_col, 3*n_row)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_umap_space_distance_calb2(
        vec_space: VectorSpace, save_name1: str, save_name2: str,
):
    feature_db = vec_space.ref_feature_db

    assert not feature_db.Ai148_flag
    cell_types = feature_db.cell_types
    calb2_p_n = reverse_dict(feature_db.cell_types)
    calb2_pos_cells_uid, calb2_neg_cells_uid = calb2_p_n[CellType.Calb2_Pos], calb2_p_n[CellType.Calb2_Neg]

    best_embedding = vec_space.get_embeddings()[0]
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
    single_plot_embedding(ax_embed, best_embedding, cell_types)
    fig_embed.set_size_inches(3, 3)
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
        p_text, _, font_kwargs = get_asterisks(p_value_pos_less_neg[cluster_id])
        ax_marginal_top.text(cluster_id+2.5, 0, p_text, **font_kwargs,
                             horizontalalignment='center', verticalalignment='center')

        p_text, _, font_kwargs = get_asterisks(p_value_neg_less_pos[cluster_id])
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



