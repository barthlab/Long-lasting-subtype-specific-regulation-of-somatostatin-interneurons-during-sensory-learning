import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict
from itertools import chain
from scipy.stats import gaussian_kde

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.feature.feature_manager import *
from src.feature.clustering import *
from src.ploter.plotting_params import *
from src.ploter.plotting_utils import *
from src.ploter.statistic_annotation import *


def plot_image_example(single_image: Image, feature_db: FeatureDataBase, single_embed: Embedding,
                       save_name: str, title: str = None, color_by_cell_type_flag: bool = False):
    num_cell, num_days = len(single_image.cells_uid), len(single_image.days)
    fig, axs = plt.subplots(num_days, num_cell, sharey='all', sharex='all')
    print(f"Cell num: {len(single_image.cells_uid)}", [x for x in single_image.cells_uid])
    print([x.name for x in single_image.days])
    if num_days == 1:
        axs = [axs]
    if num_cell == 1:
        axs = [[ax] for ax in axs]
    cell_types = feature_db.cell_types
    cluster_id = single_embed.label_by_cell

    # plot individual data
    spont_block_ratio = 0.15
    x_space = 3.3  # s
    cell_image_dict = single_image.split("cell_uid")
    for col_id, (cell_uid, cell_image) in enumerate(cell_image_dict.items()):  # type: int, (CellUID, Image)
        day_image_dict_percell = cell_image.split("day_id")
        postfix = f" {feature_db.get(CALB2_METRIC).get_cell(cell_uid):.1f}" if CALB2_METRIC in feature_db.feature_names\
            else ""
        postfix2 = f" Cluster{cluster_id[cell_uid]}"
        cell_color = CELLTYPE2COLOR[cell_types[cell_uid]] if color_by_cell_type_flag else CLUSTER_COLORLIST[cluster_id[cell_uid]]
        axs[0][col_id].set_title(cell_uid.in_short()+postfix+postfix2, color=cell_color)

        for row_id, (single_day, cell_day_image) in enumerate(day_image_dict_percell.items()):  # type: int, (SatDay|PseDay, Image)
            print(f"Row {row_id}, Col {col_id}", cell_day_image.cells_uid, cell_day_image.days, len(cell_day_image.dataset))
            tmp_ax = axs[row_id][col_id]
            single_cs = cell_day_image.dataset[0]

            x_offset = 0
            init_block = general_filter(single_cs.spont_blocks, block_type=BlockType.PreBlock)[0]  # type: SpontBlock
            tmp_ax.plot(init_block.df_f0.t_zeroed*spont_block_ratio, init_block.df_f0.v, color=cell_color, lw=0.5, alpha=0.7)
            x_offset += 15 + x_space

            for trial_id, single_trial in enumerate(general_filter(single_cs.trials, trial_type=EventType.Puff)[:5]):  # type: int, Trial
                shrink_df_f0 = single_trial.df_f0.segment(-2, 4, new_origin_t=0, relative_flag=True)
                tmp_ax.plot(shrink_df_f0.t_aligned+x_offset, shrink_df_f0.v, color=cell_color, lw=1, alpha=0.7)
                tmp_ax.scatter(x_offset, -0.5, **TRIAL_MARKER[EventType.Puff])
                x_offset += 4 + x_space

            final_block = general_filter(single_cs.spont_blocks, block_type=BlockType.PostBlock)[0]  # type: SpontBlock
            tmp_ax.plot(final_block.df_f0.t_zeroed*spont_block_ratio+x_offset, final_block.df_f0.v, color=cell_color, lw=0.5, alpha=0.7)

            tmp_ax.spines[['right', 'top',  "left", "bottom"]].set_visible(False)
            tmp_ax.set_ylim(*DISPLAY_SINGLE_DF_F0_RANGE[feature_db.exp_id])
            tmp_ax.set_xticks([])
            tmp_ax.set_yticks([0, 1], ["", r"$1\Delta F/F_0$"])
            tmp_ax.set_ylabel(single_day.name)
            tmp_ax.set_aspect(4)

    if title is not None:
        fig.suptitle(title)
    set_size(num_cell*2.5, 2*num_days, fig)
    quick_save(fig, save_name)
    plt.close(fig)


def plot_feature_example(
        save_name: str, feature_db: FeatureDataBase, feature_names: List[str], selected_days: str,
        group_of_cell_list: Dict[str, List[CellUID]], group_colors: List[str], size: Tuple[float, float]):
    n_group, n_feature = len(group_of_cell_list), len(feature_names)
    fig, axs = plt.subplots(1, n_feature, sharex='col',)

    plt.subplots_adjust(wspace=0.1)
    # collect feature_data
    for col_id, feature_name in enumerate(feature_names):
        target_feature = feature_db.get(feature_name=feature_name, day_postfix=selected_days).by_cells

        # data_min, data_max = np.min(list(target_feature.values())), np.max(list(target_feature.values()))
        # xs = np.linspace(data_min, data_max, 100)
        # xx = np.concatenate([xs, xs[::-1]])
        # xs_extend = np.concatenate([np.array([data_min, ]), xs, np.array([data_max, ])])
        tmp_ax = axs[col_id]
        for row_id, (group_name, cells_uid) in enumerate(
                group_of_cell_list.items()):  # type: int, (str, List[CellUID])
            tmp_feature_values = [target_feature[cell_uid] for cell_uid in cells_uid]
            data_min, data_max = np.min(tmp_feature_values), np.max(tmp_feature_values)
            xs = np.linspace(data_min, data_max, 100)
            xx = np.concatenate([xs, xs[::-1]])
            xs_extend = np.concatenate([np.array([data_min, ]), xs, np.array([data_max, ])])

            if len(np.unique(tmp_feature_values)) == 1:  # numpy.linalg.LinAlgError
                tmp_ax.remove()
                continue
            kde = gaussian_kde(tmp_feature_values)
            density_kde = kde(xs)
            scale_ = np.max(density_kde) * 2.5
            scaled_density = density_kde / scale_
            interp_ = np.min(scaled_density)
            scaled_density = scaled_density - interp_
            scaled_density_extend = np.concatenate([np.array([0, ]), scaled_density, np.array([0, ])])
            scatter_height = [0.5 * np.random.rand() * (kde(single_value) / scale_ - interp_)
                              for single_value in tmp_feature_values]

            yy = np.concatenate([scaled_density, np.full_like(xs, 0)])
            tmp_ax.fill(-yy+row_id, xx, facecolor=group_colors[row_id], alpha=1, edgecolor='none', zorder=2)
            # tmp_ax.plot(-scaled_density_extend + row_id, xs_extend, color='black', linewidth=0.5, zorder=3)
            tmp_ax.scatter(np.array(scatter_height) + row_id + 0.2, tmp_feature_values, facecolor='black', edgecolor='none',
                           s=6, lw=0.5, zorder=4, alpha=0.7)
            # tmp_ax.set_xlim(0, 1.1)
            tmp_ax.tick_params(axis='y', labelsize=6)
            tmp_ax.set_xlim(-0.5, n_group-0.5)
            tmp_ax.spines[['right', 'top',]].set_visible(False)
            # tmp_ax.spines['bottom'].set_linewidth(0.5)
            tmp_ax.tick_params(axis='x', which=u'both', length=0)
        tmp_ax.set_ylabel(feature_name_to_y_axis_label(feature_name))
        tmp_ax.set_xticks(np.arange(len(group_of_cell_list)), list(group_of_cell_list.keys()), )

    set_size(size[0] * n_feature, size[1], fig)
    fig.tight_layout()
    quick_save(fig, save_name)

