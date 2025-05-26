import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict, Counter
from scipy.spatial import distance
from scipy.stats import gaussian_kde

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


def plot_labeling_quality_summary(vec_space: VectorSpace, save_name: str,
                                  embed_dict: Dict[Labelling, List[Embedding]]):
    fig, ax = plt.subplots(1, 1)
    all_occurrences = []
    for single_labelling, embed_list in embed_dict.items():
        avg_score = np.mean([single_embed.score for single_embed in embed_list])
        n_cluster = single_labelling.n_cluster
        occurrences = 100 * len(embed_list) / vec_space.n_embeddings
        if n_cluster in PLOTTING_CLUSTERS_OPTIONS:
            ax.scatter(avg_score, occurrences, facecolor='none', edgecolor=CLUSTER_COLORLIST[n_cluster],
                       s=5, alpha=0.7)
            all_occurrences.append(occurrences)

    ax.axhline(y=np.percentile(all_occurrences, q=TOP_LABELLING_OCCURRENCE_THRESHOLD),
               lw=1, ls='--', alpha=0.7, color='gray', zorder=-2)

    ax.spines[['right', 'top']].set_visible(False)
    legend_handles = []
    for n_cluster_option in PLOTTING_CLUSTERS_OPTIONS:
        legend_handles.append(plt.Line2D(
            [0], [0], marker='o', linestyle="None", alpha=0.7,
            label=f"{n_cluster_option} clusters",
            markerfacecolor="none", markeredgecolor=CLUSTER_COLORLIST[n_cluster_option], ))
    ax.legend(handles=legend_handles, frameon=False, loc='upper left')
    ax.set_xlabel("Score")
    ax.set_ylabel("Occurrence (%)")

    fig.set_size_inches(2, 2)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_feature_summary(
        save_name: str, feature_db: FeatureDataBase, feature_names: List[str], selected_days: str,
        group_of_cell_list: Dict[str, List[CellUID]], group_colors: List[str], size: Tuple[float, float]):
    n_group, n_feature = len(group_of_cell_list), len(feature_names)
    fig, axs = plt.subplots(n_group, n_feature, sharex='col', sharey='all')
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    # collect feature_data
    for col_id, feature_name in enumerate(feature_names):
        target_feature = feature_db.get(feature_name=feature_name, day_postfix=selected_days).by_cells

        data_min, data_max = np.min(list(target_feature.values())), np.max(list(target_feature.values()))
        xs = np.linspace(data_min, data_max, 100)
        xx = np.concatenate([xs, xs[::-1]])
        xs_extend = np.concatenate([np.array([data_min, ]), xs, np.array([data_max, ])])
        for row_id, (group_name, cells_uid) in enumerate(group_of_cell_list.items()):  # type: int, (str, List[CellUID])
            tmp_ax = axs[row_id, col_id]

            tmp_feature_values = [target_feature[cell_uid] for cell_uid in cells_uid]
            if len(np.unique(tmp_feature_values)) == 1:  # numpy.linalg.LinAlgError
                tmp_ax.remove()
                continue
            kde = gaussian_kde(tmp_feature_values)
            density_kde = kde(xs)
            scaled_density = density_kde / np.max(density_kde)
            scaled_density_extend = np.concatenate([np.array([0, ]), scaled_density, np.array([0, ])])

            yy = np.concatenate([scaled_density, np.full_like(xs, 0)])
            tmp_ax.fill(xx, yy, facecolor=group_colors[row_id], alpha=1, edgecolor='none', zorder=2)
            tmp_ax.plot(xs_extend, scaled_density_extend, color='black', linewidth=0.5,  zorder=3)
            tmp_ax.set_ylim(0, 1.1)
            tmp_ax.set_yticks([])
            tmp_ax.spines[['right', 'top', 'left']].set_visible(False)
            tmp_ax.spines['bottom'].set_linewidth(0.5)
            if col_id == 0:
                tmp_ax.set_ylabel(group_name, rotation=0, va="center", ha='right')
            if row_id == n_group - 1:
                tmp_ax.set_xticks([])
                tmp_ax.set_xlabel(f"#{col_id+1}")
            if row_id == 0:
                title_color = FEATURE_LABEL2COLOR[feature_name_to_label(feature_name)]
                rect = plt.Rectangle((0.1, 1.2), 0.8, 0.5, clip_on=False, linewidth=0,
                                     facecolor=title_color, edgecolor='none', transform=tmp_ax.transAxes, )
                tmp_ax.add_patch(rect)

    fig.set_size_inches(size[0]*n_feature, size[1]*n_group + size[1]*0.5)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_fold_change_bars(
        save_name: str, feature_db: FeatureDataBase, bars_list: List[str], size: Tuple[float, float],
        group_of_cell_list: Dict[str, List[CellUID]], group_colors: List[str]):
    n_group, n_bar = len(group_of_cell_list), len(bars_list)

    bar_offset, bar_width = 1/(n_group+1), 2*0.4/(n_group+1)
    fig, ax = plt.subplots(1, 1)

    for group_id, (group_name, cells_uid) in enumerate(
            group_of_cell_list.items()):  # type: int, (str, List[CellUID])
        statistic_dict = {}
        for bar_id, bar_name in enumerate(bars_list):
            tmp_fold_change = feature_db.get(feature_name=FOLD_CHANGE_ACC456_FEATURE, day_postfix=bar_name)

            tmp_data = {cell_uid: tmp_fold_change.by_cells[cell_uid] for cell_uid in cells_uid}
            if DEBUG_FLAG:
                print(group_name, bar_name, nan_mean(list(tmp_data.values())), nan_sem(list(tmp_data.values())))
            oreo_bar(ax, list(tmp_data.values()), x_position=bar_id+group_id*bar_offset,
                     width=bar_width, color=group_colors[group_id])
            statistic_dict[bar_id+group_id*bar_offset] = tmp_data
        paired_ttest_with_Bonferroni_correction(ax, statistic_dict, simple_flag=False)
    ax.set_ylim(0, 2.4)
    ax.set_yticks([0., 0.5, 1., 1.5, 2.])
    ax.spines[['right', 'top']].set_visible(False)
    ax.axvspan(0.75, len(bars_list) - 0.5, lw=0, alpha=0.4, zorder=0,
               color=OTHER_COLORS['SAT'] if feature_db.SAT_flag else OTHER_COLORS["PSE"], )
    ax.set_ylabel(r'Peak response (normalized)', fontsize=7)
    ax.set_xticks(np.arange(len(bars_list))+bar_offset*((n_group-1)/2), bars_list)
    ax.axhline(y=1, lw=1, color='black', alpha=0.4, ls='--')
    ax.tick_params(axis='x', which=u'both', length=0)

    fig.set_size_inches(*size)
    fig.tight_layout()
    quick_save(fig, save_name)
