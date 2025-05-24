import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict
from itertools import chain
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.transforms as transforms

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
        feature_db: FeatureDataBase, save_name: str, feature_name: str, sorted_features_names: List[str],
        selected_days: str = "ACC456", calb2_func=CALB2_RESIZE_FUNC,
):
    assert not feature_db.Ai148_flag

    target_feature = feature_db.get(feature_name=feature_name, day_postfix=selected_days).by_cells
    feature_sorted_id = sorted_features_names.index(feature_name)
    cell_types = reverse_dict(feature_db.cell_types)
    calb2_value = {k: calb2_func(v) for k, v in feature_db.get('Calb2 Mean').by_cells.items()}
    if np.sum(np.isnan(list(target_feature.values()))) > 0:
        raise ValueError(f"Found nan value in target_feature: {feature_name}")

    fig, axs = plt.subplots(2, 1)
    ax = axs[1]
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
    axs[0].text(0, 0, p_text, ha='center', va='center', color='black',
                alpha=font_kwargs['alpha'], fontsize=8)
    ax.axvspan(calb2_func(CALB2_THRESHOLD), top_right_corner[0],
               color=CELLTYPE2COLOR[CellType.Calb2_Pos], alpha=0.1, lw=0)

    ax.scatter(*top_right_corner, s=0, alpha=0)
    ax.spines[['right', 'top']].set_visible(False)
    axs[0].spines[['left', 'bottom', 'right', 'top']].set_visible(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    ax.set_xticks([calb2_func(_v) for _v in CALB2_TICKS], CALB2_TICKS)
    ax.set_xlabel("Calb2 Intensity (A.U.)")
    ax.set_ylabel(feature_name_to_y_axis_label(feature_name))
    ax.set_title(feature_name_to_title_short(feature_name), fontsize=8)

    fig.set_size_inches(1.8, 3.2)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_feature_distribution_calb2(
        feature_db: FeatureDataBase, save_name: str, sorted_p_value_dict: Dict[str, float], top_k: int = 30,
        period_name_flag: bool = False
):
    assert not feature_db.Ai148_flag
    sorted_features_names = list(sorted_p_value_dict.keys())
    print(sorted_features_names)
    sorted_features_pvalues = list(sorted_p_value_dict.values())
    n_feature = len(sorted_features_pvalues)

    if period_name_flag:
        sorted_features_labels = [feature_name_to_period_name(single_feature_name)
                                  for single_feature_name in sorted_features_names]
        sorted_features_colors = [PERIOD_NAME2COLOR[single_label]
                                  for single_label in sorted_features_labels]
    else:
        sorted_features_labels = [feature_name_to_label(single_feature_name)
                                  for single_feature_name in sorted_features_names]
        sorted_features_colors = [FEATURE_LABEL2COLOR[single_label]
                                  for single_label in sorted_features_labels]

    fig, axs = plt.subplots(2, 1)
    ax = axs[1]
    bar_pos = [i for i in range(top_k)] + [i+1 for i in range(top_k, n_feature)]
    bars = ax.bar(bar_pos, sorted_features_pvalues, width=1, edgecolor='white', lw=0.5,
                  color=sorted_features_colors, log=True)
    legend_handles = []
    for label_name, label_color in PERIOD_NAME2COLOR.items() if period_name_flag else FEATURE_LABEL2COLOR.items():
        patch = mpatches.Patch(color=label_color, label=label_name)
        legend_handles.append(patch)

    axs[0].legend(handles=legend_handles, title="Periods" if period_name_flag else "Features",
                  frameon=False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', alpha=0.3)
    xticks = [0, ] + [10*(i+1) - 1 for i in range(int(n_feature/10))]
    xticks_corrected = [i if i < top_k else i+1 for i in xticks]
    xticklabels = [f"#{i+1}" if i < top_k else f"#{i}" for i in xticks_corrected]
    ax.set_xticks(xticks_corrected, xticklabels)
    ax.axhline(y=SIGNIFICANT_P, lw=1, color='black', ls='--', alpha=0.8)
    x_min, _ = ax.get_xlim()
    ax.text(x_min - 1.5, SIGNIFICANT_P, r"$\ast$" + f" {SIGNIFICANT_P}",
            ha='right', va='center', color='black')
    ax.set_xlabel("Features")
    ax.set_ylabel("p-value")

    kth_pvalue = sorted_features_pvalues[top_k - 1]
    # top-k feature highlight
    # y_min, _ = ax.get_ylim()
    # rect_x, rect_y = -0.5, y_min * TEXT_OFFSET_SCALE
    # rect_width = top_k
    # rect_height = (kth_pvalue - y_min) * TEXT_OFFSET_SCALE
    # rect = mpatches.Rectangle((rect_x, rect_y), rect_width, rect_height,
    #                           linewidth=1, edgecolor='red', facecolor='none',
    #                           linestyle='--', clip_on=False, zorder=5)
    # ax.add_patch(rect)

    #
    annot_text = f"Top {top_k} Features\np <= {kth_pvalue:.2e}"
    ax.text(top_k * 0.5, kth_pvalue * 15, annot_text, va="bottom", ha='center')
    # ax.plot([0, top_k], [kth_pvalue, kth_pvalue], color='red', lw=2, ls='--')
    ax.axvline(x=top_k, ymax=0.7, color='red', lw=1., ls='--')
    # annot_xy = (top_k * 0.75, kth_pvalue * TEXT_OFFSET_SCALE)
    # annot_xytext = (top_k * 0.25, kth_pvalue * 15)
    # ax.annotate(annot_text, xy=annot_xy, xytext=annot_xytext,
    #             arrowprops=dict(facecolor='black', shrink=0.02, width=0.5, headwidth=2,# headlength=1,
    #                             connectionstyle="arc3,rad=.2"),
    #             bbox=dict(boxstyle="round,pad=0.4", fc="ivory", ec="black", lw=1, alpha=0.8))

    fig.set_size_inches(7.5, 5)
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
                           facecolors=colors[dot_i], edgecolors='white', lw=0.1, label=labels[dot_i])

        kde_x = stats.gaussian_kde(dots[:, 0])
        ax_hist_x.fill_between(x_range, kde_x(x_range), color=colors[dot_i], alpha=0.3, lw=0)

        kde_y = stats.gaussian_kde(dots[:, 1])
        ax_hist_y.fill_betweenx(y_range, kde_y(y_range), color=colors[dot_i], alpha=0.3, lw=0)

    ax_scatter.set_xlabel('Euclidean distance')
    ax_scatter.set_ylabel('Chebyshev distance')
    # ax_scatter.grid(True, lw=0.5, ls='--', alpha=0.7, zorder=-2)
    # ax_scatter.legend(frameon=False, fontsize=5, loc="lower right")
    ax_scatter.spines[['right', 'top']].set_visible(False)

    ax_hist_x.spines[['right', 'top', 'left']].set_visible(False)
    ax_hist_x.set_yticks([])
    ax_hist_y.spines[['right', 'top', 'bottom']].set_visible(False)
    ax_hist_y.set_xticks([])

    fig.set_size_inches(4, 4)
    fig.tight_layout()
    quick_save(fig, save_name1)
    plt.close(fig)

    for i, distance_name in zip([0, 1], ['Euclidean distance', 'Chebyshev distance']):
        fig, ax = plt.subplots(1, 1)
        for dot_i, dots in enumerate(all_dots):
            oreo_bar(ax, dots[:, 0], x_position=dot_i, width=0.6, color=colors[dot_i], label=labels[dot_i])

        y_level = np.max([np.mean(dots[:, 0]) for dots in all_dots]) + 2
        for dot_i in range(3):
            for dot_j in range(dot_i+1, 3):
                _, p_value = stats.mannwhitneyu(all_dots[dot_i][:, i], all_dots[dot_j][:, i])
                statistic_bar(ax, dot_i, dot_j, y_level, p_value)
                y_level += 1.5

        ax.spines[['right', 'top',]].set_visible(False)
        # ax.legend(title="Pairs", frameon=False, fontsize=4, title_fontsize=4)
        ax.set_ylabel(distance_name)
        ax.set_xticks([])

        fig.set_size_inches(2, 2)
        fig.tight_layout()
        quick_save(fig, save_name2+f"{distance_name}.png")
        plt.close(fig)


def plot_feature_hierarchy_structure(
        feature_db: FeatureDataBase, save_name: str, feature_names: List[str], sorted_p_value_dict: Dict[str, float],
        selected_days: str = "ACC456",
):
    if feature_db.Ai148_flag:
        sorted_cells_uid = feature_db.cells_uid
    else:
        calb2_mean = feature_db.get("Calb2 Mean").by_cells
        cell_types = feature_db.cell_types
        sorted_cells_uid = sorted(list(calb2_mean.keys()), key=lambda cell_uid: calb2_mean[cell_uid], reverse=True)
        fig, ax = plt.subplots(1, 1)
        for row_id, cell_uid in enumerate(sorted_cells_uid):
            ax.barh(-row_id, CALB2_RESIZE_FUNC(calb2_mean[cell_uid]), color=CELLTYPE2COLOR[cell_types[cell_uid]], lw=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'left', 'bottom', 'top']].set_visible(False)
        fig.set_size_inches(0.5, 1)
        fig.tight_layout()
        quick_save(fig, save_name + "_calb2.png")

    all_features = {}
    for single_feature_name in feature_names:
        tmp_feature_by_cells = feature_db.get(feature_name=single_feature_name, day_postfix=selected_days).standardized().by_cells
        feature_vector = np.array([tmp_feature_by_cells[cell_uid] for cell_uid in sorted_cells_uid])
        all_features[single_feature_name] = feature_vector
    feature_matrix = np.array(list(all_features.values()))
    n_feature, n_cell = feature_matrix.shape

    distance_matrix = pdist(feature_matrix, metric='euclidean')
    linked = linkage(distance_matrix, method='ward')

    fig, axss = plt.subplots(3, 2,  height_ratios=[0.6, 1, 0.8], width_ratios=[1, 0.02], sharex='col')
    plt.subplots_adjust(hspace=0)
    axs = axss[:, 0]
    dendro_info = dendrogram(linked, ax=axs[0], orientation='top',
                             distance_sort='descending', show_leaf_counts=True)
    axs[0].set_title('Hierarchical Dendrogram')
    reordered_feature_names = [feature_names[i] for i in dendro_info['leaves']]
    reordered_feature_color = [FEATURE_LABEL2COLOR[feature_name_to_label(single_feature)]
                               for single_feature in reordered_feature_names]

    axs[0].spines[['right', 'top', 'bottom']].set_visible(False)
    axs[0].tick_params(axis='x', which='both', length=0, labelbottom=False, )
    # axs[0].set_xlabel('Features')
    axs[0].set_ylabel('Distance (Ward)')

    xticks_locations = axs[0].get_xticks()
    bar_height = [sorted_p_value_dict[single_feature] for single_feature in reordered_feature_names]
    bar_color = [FEATURE_LABEL2COLOR[feature_name_to_label(single_feature)]
                 for single_feature in reordered_feature_names]

    axss[0, 1].remove()
    axss[2, 1].remove()

    reordered_features = feature_matrix[dendro_info['leaves'], :]
    print(reordered_features.shape)
    sns.heatmap(reordered_features.T.repeat(10, axis=1), vmin=-5, vmax=5,
                ax=axs[1], cmap='bwr', cbar_ax=axss[1, 1], cbar_kws={'label': 'Normalized Feature Value'})
    axs[1].set_ylabel("Cells")
    axs[1].set_yticks([])
    axs[1].tick_params(axis='x', which='both', length=0, labelbottom=False, )

    blended_transform = transforms.blended_transform_factory(axs[2].transData, axs[2].transAxes)
    for xtick, pvalue in zip(xticks_locations, bar_height):
        if pvalue > SIGNIFICANT_P_EXTRA:
            continue
        rect = mpatches.Rectangle(
            (xtick-5, 1.1), 10, 0.05, lw=0, zorder=3,
            edgecolor='none', facecolor='black', transform=blended_transform)
        axs[2].add_patch(rect)
        rect.set_clip_on(False)

    bars = axs[2].bar(xticks_locations, bar_height,
                      width=10, edgecolor='white', lw=1,
                      color=bar_color, log=True)
    axs[2].invert_yaxis()

    axs[2].spines[['right', 'left', 'bottom']].set_visible(False)
    axs[2].yaxis.grid(color='gray', alpha=0.3)
    # axs[2].tick_params(axis='x', which='both', length=0, labeltop=False,)
    # axs[2].axhline(y=SIGNIFICANT_P, lw=1, color='black', ls='--', alpha=0.8)
    axs[2].axhline(y=SIGNIFICANT_P_EXTRA, lw=1, color='black', ls='--', alpha=0.8)

    x_min, x_max = axs[1].get_xlim()
    # axs[2].text(x_max + 2.5, SIGNIFICANT_P, r"$\ast$" + f" {SIGNIFICANT_P}",
    #             ha='left', va='center', color='black')
    axs[2].text(x_max + 2.5, SIGNIFICANT_P_EXTRA, r"$\ast$"*2 + f" {SIGNIFICANT_P_EXTRA}",
                ha='left', va='center', color='black')
    axs[2].set_ylabel("p-value")

    axs[2].tick_params(axis='x', which='both', length=0, )
    axs[2].set_xticks(xticks_locations, reordered_feature_names, rotation=90, fontsize=4)
    for i, xtick_label in enumerate(axs[2].xaxis.get_ticklabels()):
        xtick_label.set_color(reordered_feature_color[i])

    fig.set_size_inches(8., 4)
    fig.tight_layout()
    quick_save(fig, save_name+".png")
    # plt.show()