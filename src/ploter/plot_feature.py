import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict
from itertools import chain
from scipy import stats

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


def plot_single_feature_Calb2(
        feature_db: FeatureDataBase, save_name: str, feature_name: str,
        selected_days: str = "ACC456", calb2_func=lambda x: np.log10(x - CALB2_MINIMAL),
):
    assert not feature_db.Ai148_flag

    target_feature = feature_db.get(feature_name=feature_name, day_postfix=selected_days).by_cells
    cell_types = reverse_dict(feature_db.cell_types)
    calb2_value = {k: calb2_func(v) for k, v in feature_db.get('Calb2 Mean').by_cells.items()}
    if np.sum(np.isnan(list(target_feature.values()))) > 0:
        raise ValueError(f"Found nan value in target_feature: {feature_name}")

    fig, ax = plt.subplots(1, 1)

    for cell_uid in cell_types[CellType.Calb2_Pos]:
        ax.scatter(calb2_value[cell_uid], target_feature[cell_uid],
                   s=3, color=CELLTYPE2COLOR[CellType.Calb2_Pos], alpha=0.7)
    for cell_uid in cell_types[CellType.Calb2_Neg]:
        ax.scatter(calb2_value[cell_uid], target_feature[cell_uid],
                   s=3, color=CELLTYPE2COLOR[CellType.Calb2_Neg], alpha=0.7)

    p_text, _, font_kwargs = get_asterisks(
        p_value=stats.ttest_ind(
            [target_feature[cell_uid] for cell_uid in cell_types[CellType.Calb2_Pos]],
            [target_feature[cell_uid] for cell_uid in cell_types[CellType.Calb2_Neg]],
        ).pvalue, double_line_flag=False)

    top_right_corner = (TEXT_OFFSET_SCALE * nan_max(list(calb2_value.values())),
                        TEXT_OFFSET_SCALE * nan_max(list(target_feature.values())))
    ax.text(*top_right_corner, p_text, ha='right', va='top', color='black',
            alpha=font_kwargs['alpha'], fontsize=8)
    ax.axvspan(calb2_func(CALB2_THRESHOLD), top_right_corner[0],
               color=CELLTYPE2COLOR[CellType.Calb2_Pos], alpha=0.1, lw=0)

    ax.scatter(*top_right_corner, s=0, alpha=0)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([calb2_func(_v) for _v in CALB2_TICKS], CALB2_TICKS)
    ax.set_xlabel("Calb2 Intensity (A.U.)")
    ax.set_ylabel(feature_name_to_y_axis_label(feature_name))
    ax.set_title(feature_name_to_title_short(feature_name), fontsize=8)

    fig.set_size_inches(1.8, 1.8)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_feature_distribution_calb2(
        feature_db: FeatureDataBase, save_name: str, sorted_p_value_dict: Dict[str, float], top_k: int = 25,
):
    assert not feature_db.Ai148_flag
    sorted_features_names = list(sorted_p_value_dict.keys())
    print(sorted_features_names)
    sorted_features_pvalues = list(sorted_p_value_dict.values())
    sorted_features_labels = [feature_name_to_label(single_feature_name)
                              for single_feature_name in sorted_features_names]
    n_feature = len(sorted_features_pvalues)
    sorted_features_colors = [FEATURE_LABEL2COLOR[single_label]
                              for single_label in sorted_features_labels]

    fig, ax = plt.subplots(1, 1)
    bars = ax.bar(range(n_feature), sorted_features_pvalues,
                  color=sorted_features_colors, log=True)
    legend_handles = []
    for label_name, label_color in FEATURE_LABEL2COLOR.items():
        patch = mpatches.Patch(color=label_color, label=label_name)
        legend_handles.append(patch)

    ax.legend(handles=legend_handles, title="Features", frameon=False, fontsize=4, title_fontsize=4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray')
    ax.set_xticks([])
    ax.axhline(y=SIGNIFICANT_P, lw=1, color='black', ls='--', alpha=0.5)
    x_min, _ = ax.get_xlim()
    ax.text(x_min - 1.5, SIGNIFICANT_P, r"$\ast$" + f" {SIGNIFICANT_P}",
            ha='right', va='center', color='black')
    ax.set_xlabel("Features")
    ax.set_ylabel("p-value")

    # top-k feature highlight
    y_min, _ = ax.get_ylim()
    kth_pvalue = sorted_features_pvalues[top_k - 1]
    rect_x, rect_y = -0.5, y_min * TEXT_OFFSET_SCALE
    rect_width = top_k
    rect_height = (kth_pvalue - y_min) * TEXT_OFFSET_SCALE
    rect = mpatches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                              linewidth=1, edgecolor='red', facecolor='none',
                              linestyle='--', clip_on=False, zorder=5)
    ax.add_patch(rect)
    annot_text = f"Top {top_k} Features\np <= {kth_pvalue:.2e}"
    annot_xy = (top_k * 0.75, kth_pvalue * TEXT_OFFSET_SCALE)
    annot_xytext = (top_k * 0.75 + n_feature * 0.05, kth_pvalue * 5)
    ax.annotate(annot_text, xy=annot_xy, xytext=annot_xytext,
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4,
                                connectionstyle="arc3,rad=.2"),
                bbox=dict(boxstyle="round,pad=0.4", fc="ivory", ec="black", lw=1, alpha=0.8))

    fig.set_size_inches(7.5, 2.5)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_vector_space_distance_calb2(
        vec_space: VectorSpace, save_name1: str, save_name2: str,
):
    feature_db = vec_space.ref_feature_db

    assert not feature_db.Ai148_flag
    cell_types = reverse_dict(feature_db.cell_types)

    dots_pvp = vec_space.pair_distances(cell_types[CellType.Calb2_Pos])
    dots_pvn = vec_space.pair_distances(cell_types[CellType.Calb2_Pos], cell_types[CellType.Calb2_Neg])
    dots_nvn = vec_space.pair_distances(cell_types[CellType.Calb2_Neg])

    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row',
                            width_ratios=[4, 1], height_ratios=[1, 4], )
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    ax_scatter, ax_hist_x, ax_hist_y = axs[1, 0], axs[0, 0], axs[1, 1]
    axs[0, 1].remove()

    all_dots = [dots_pvn, dots_pvp, dots_nvn]
    # colors = ["#DF1EE1", "#E1DF1E", "#1EE1DF"]
    colors = ['#FF7F0E', '#0EFF7F', '#7F0EFF']
    labels = ["Inter. (SST-Calb2, SST-O)", "Intra. SST-Calb2", "Intra. SST-O"]

    x_concat = np.concatenate([dots[:, 0] for dots in all_dots])
    x_range = np.linspace(0, np.max(x_concat) * 1.1, 200)
    y_concat = np.concatenate([dots[:, 1] for dots in all_dots])
    y_range = np.linspace(0, np.max(y_concat) * 1.1, 200)

    for dot_i, dots in enumerate(all_dots):
        ax_scatter.scatter(dots[:, 0], dots[:, 1], s=2, alpha=0.7,
                           color=colors[dot_i], label=labels[dot_i])

        kde_x = stats.gaussian_kde(dots[:, 0])
        ax_hist_x.fill_between(x_range, kde_x(x_range), color=colors[dot_i], alpha=0.3, lw=0)

        kde_y = stats.gaussian_kde(dots[:, 1])
        ax_hist_y.fill_betweenx(y_range, kde_y(y_range), color=colors[dot_i], alpha=0.3, lw=0)

    ax_scatter.set_xlabel('Euclidean Distance')
    ax_scatter.set_ylabel('Chebyshev Distance')
    # ax_scatter.grid(True, lw=0.5, ls='--', alpha=0.7, zorder=-2)
    ax_scatter.legend(frameon=False, fontsize=5, loc="lower right")
    ax_scatter.spines[['right', 'top']].set_visible(False)

    ax_hist_x.spines[['right', 'top', 'left']].set_visible(False)
    ax_hist_x.set_yticks([])
    ax_hist_y.spines[['right', 'top', 'bottom']].set_visible(False)
    ax_hist_y.set_xticks([])

    fig.set_size_inches(3, 3)
    fig.tight_layout()
    quick_save(fig, save_name1)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    for dot_i, dots in enumerate(all_dots):
        oreo_bar(ax, dots[:, 0], x_position=dot_i, width=0.6, color=colors[dot_i], label=labels[dot_i])

    y_level = np.max([np.mean(dots[:, 0]) for dots in all_dots]) + 2
    for dot_i in range(3):
        for dot_j in range(dot_i+1, 3):
            _, p_value = stats.mannwhitneyu(all_dots[dot_i][:, 0], all_dots[dot_j][:, 0])
            statistic_bar(ax, dot_i, dot_j, y_level, p_value)
            y_level += 1.5

    ax.spines[['right', 'top',]].set_visible(False)
    # ax.legend(title="Pairs", frameon=False, fontsize=4, title_fontsize=4)
    ax.set_ylabel('Euclidean Distance')
    ax.set_xticks([])

    fig.set_size_inches(1.5, 3)
    fig.tight_layout()
    quick_save(fig, save_name2)
    plt.close(fig)


