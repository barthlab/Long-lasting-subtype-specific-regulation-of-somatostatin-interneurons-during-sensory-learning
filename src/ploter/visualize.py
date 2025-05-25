import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.feature.feature_manager import *
from src.feature.clustering_params import *
from src.ploter.plotting_params import *
from src.ploter.plotting_utils import *


def plot_cell_session(single_cs: CellSession, save_name: str):
    fig, axs = plt.subplots(1, 3, width_ratios=[3, 1, 1])
    for stim_time, stim_label in zip(single_cs.stims.t, single_cs.stims.label):
        axs[0].axvspan(xmin=stim_time, xmax=stim_time + AP_DURATION,
                       color=EVENT2COLOR[stim_label], alpha=0.2)
    axs[0].set_ylim(-1, 3.0)
    axs[0].plot(single_cs.df_f0.t, single_cs.df_f0.v, lw=1, alpha=0.7, color='black')
    # axs[0].fill_between(single_cs.df_f0.t, np.zeros_like(single_cs.df_f0.drop), single_cs.df_f0.drop,
    #                     lw=0, alpha=0.2, color='red')

    for trial_id, single_trial in enumerate(single_cs.trials):  # type: int, Trial
        ls = '--' if single_trial.drop_flag else "-"
        axs[1].plot(single_trial.df_f0.t_aligned, single_trial.df_f0.v + trial_id * DY_DF_F0,
                    color=EVENT2COLOR[single_trial.trial_type], alpha=0.7, lw=1, ls=ls)
        axs[1].add_patch(mpatches.Rectangle(
            (0, (trial_id - 0.4) * DY_DF_F0), AP_DURATION, 0.8*DY_DF_F0,
            facecolor=EVENT2COLOR[single_trial.trial_type], alpha=0.3, edgecolor='none', ))

    for block_order, single_block in enumerate(single_cs.spont_blocks):
        axs[2].plot(single_block.df_f0.t_zeroed, single_block.df_f0.v + (block_order-0.5) * DY_DF_F0,
                    color=EVENT2COLOR[single_block.block_type], alpha=0.7, lw=1)
        axs[0].add_patch(mpatches.Rectangle(
            (single_block.block_start, -1.5 * DY_DF_F0), single_block.block_len, DY_DF_F0,
            facecolor=EVENT2COLOR[single_block.block_type], alpha=0.3, edgecolor='none', ))
    axs[2].sharey(axs[1])
    for i in range(3):
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].set_xlabel("Time [s]")
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[0].set_ylabel(r"$\Delta F/F_0$")
    axs[1].set_ylabel("Trials")
    axs[2].set_ylabel("Spont. Blocks")

    fig.suptitle(single_cs.__repr__())
    fig.set_size_inches(16, 5)
    fig.tight_layout()
    quick_save(save_name=save_name, fig=fig)


def plot_image(single_image: Image, feature_db: FeatureDataBase, save_name: str, title: str = None):
    num_cell, num_days = len(single_image.cells_uid), len(single_image.days)
    fig, axs = plt.subplots(num_days + 1, 3 * num_cell, width_ratios=[1, 1, 2]*num_cell,
                            sharey='row', sharex='col')
    print(f"Cell num: {len(single_image.cells_uid)}", [x for x in single_image.cells_uid])
    print([x.name for x in single_image.days])
    cell_types = feature_db.cell_types

    # plot individual data
    max_depth = 0
    cell_image_dict = single_image.split("cell_uid")
    for col_id, (cell_uid, cell_image) in enumerate(cell_image_dict.items()):  # type: int, (CellUID, Image)
        day_image_dict_percell = cell_image.split("day_id")
        postfix = f" {feature_db.get(CALB2_METRIC).get_cell(cell_uid):.1f}" if CALB2_METRIC in feature_db.feature_names\
            else ""
        axs[0, col_id * 3 + 1].set_title(cell_uid.in_short()+postfix, color=CELLTYPE2COLOR[cell_types[cell_uid]])

        for row_id, (single_day, cell_day_image) in enumerate(day_image_dict_percell.items()):  # type: int, (SatDay|PseDay, Image)
            print(f"Row {row_id}, Col {col_id}", cell_day_image.cells_uid, cell_day_image.days, len(cell_day_image.dataset))
            puff_ax, blank_ax, spont_ax = axs[row_id, col_id*3], axs[row_id, col_id*3+1], axs[row_id, col_id*3+2]
            # plot one cell's one day
            session_dy = 0
            for single_cs in cell_day_image.dataset:
                max_depth = min(-(len(single_cs.trials) + 2) * DY_DF_F0 - session_dy, max_depth)
                for trial_id, single_trial in enumerate(single_cs.trials):  # type: int, Trial
                    ls = '--' if single_trial.drop_flag else "-"
                    tmp_ax = puff_ax if single_trial.trial_type is EventType.Puff else blank_ax
                    tmp_ax.plot(single_trial.df_f0.t_aligned,
                                single_trial.df_f0.v - trial_id * DY_DF_F0 - session_dy,
                                color=EVENT2COLOR[single_trial.trial_type], alpha=0.7, lw=1, ls=ls)
                for block_order, single_block in enumerate(single_cs.spont_blocks):
                    spont_ax.plot(single_block.df_f0.t_zeroed,
                                  single_block.df_f0.v - (block_order - 0.5) * DY_DF_F0 - session_dy,
                                  color=EVENT2COLOR[single_block.block_type], alpha=0.7, lw=1)
                session_dy += (len(single_cs.trials) + 2) * DY_DF_F0
            puff_ax.axvspan(0, AP_DURATION, color=OTHER_COLORS["puff"], alpha=0.1, lw=0)
            # blank_ax.axvspan(0, AP_DURATION, color=EVENT2COLOR[EventType.Blank], alpha=0.2, lw=0)

            puff_ax.spines[['right', 'top']].set_visible(False)
            blank_ax.spines[['right', 'top', "left"]].set_visible(False)
            spont_ax.spines[['right', 'top', "left"]].set_visible(False)

            puff_ax.set_yticks([0, 1], ["", r"$1\Delta F/F_0$"])
            puff_ax.set_xticks([0, 1])
            blank_ax.set_xticks([0, 1])
            puff_ax.set_ylabel(single_day.name)

        # plot average data
        avg_puff_ax, avg_blank_ax, avg_spont_ax = axs[-1, col_id*3], axs[-1, col_id*3+1], axs[-1, col_id*3+2]

        all_trials = cell_image.trials
        oreo(ax=avg_puff_ax,
             list_timeseries=[single_trial.df_f0 for single_trial in
                              general_filter(all_trials, trial_type=EventType.Puff)],
             mean_kwargs={"color": EVENT2COLOR[EventType.Puff], "alpha": 0.7, "lw": 1},
             fill_kwargs={"color": EVENT2COLOR[EventType.Puff], "alpha": 0.2, "lw": 0})
        avg_puff_ax.axvspan(0, AP_DURATION, color=OTHER_COLORS["puff"], alpha=0.1, lw=0)

        oreo(ax=avg_blank_ax,
             list_timeseries=[single_trial.df_f0 for single_trial in
                              general_filter(all_trials, trial_type=EventType.Blank)],
             mean_kwargs={"color": EVENT2COLOR[EventType.Blank], "alpha": 0.7, "lw": 1},
             fill_kwargs={"color": EVENT2COLOR[EventType.Blank], "alpha": 0.2, "lw": 0})
        avg_puff_ax.spines[['right', 'top']].set_visible(False)
        avg_blank_ax.spines[['right', 'top', "left"]].set_visible(False)
        avg_spont_ax.spines[['right', 'top', "left"]].set_visible(False)
        avg_puff_ax.set_yticks([0, 1], ["", r"$1\Delta F/F_0$"])
        avg_puff_ax.set_xticks([0, 1])
        avg_blank_ax.set_xticks([0, 1])
        avg_spont_ax.set_xticks([0, 60])
        avg_puff_ax.set_ylabel("average")
        avg_blank_ax.set_xlabel("Time [s]")

    for row_id in range(num_days):
        for col_id in range(3*num_cell):
            axs[row_id, col_id].set_ylim(max_depth, DISPLAY_SINGLE_DF_F0_RANGE[single_image.exp_id][1])

    if title is not None:
        fig.suptitle(title)
    fig.set_size_inches(num_cell*6, 1.5*(num_days+1))
    fig.tight_layout()
    if FIGURE_SHOW_FLAG:
        plt.show()
    else:
        save_path = path.join(ROOT_PATH, FIGURE_PATH, save_name)
        os.makedirs(path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_plasticity_manifold(features: FeatureDataBase, days1: str, days2: str, save_name: str):
    def dot_resize(x):
        return np.log(x)
    evoked_peak_control = features.get(EVOKED_RESPONSE_FEATURE, day_postfix=days1)
    evoked_peak_test = features.get(EVOKED_RESPONSE_FEATURE, day_postfix=days2)

    cells_types = features.cell_types
    calb2_mean = features.get("Calb2 Mean")
    n_cell = len(cells_types)

    fig3d, ax3d = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
    fig2d, ax2d = plt.subplots(1, 1)
    calb2_intensity = [CALB2_RESIZE_FUNC(calb2_mean.get_cell(cell_uid)) for cell_uid in features.cells_uid]
    dots = [
        [dot_resize(evoked_peak_test.get_cell(cell_uid)) for cell_uid in features.cells_uid],
        [dot_resize(evoked_peak_control.get_cell(cell_uid)) for cell_uid in features.cells_uid],
        calb2_intensity,
    ]
    dot_colors = [CELLTYPE2COLOR[cells_types[cell_uid]] for cell_uid in features.cells_uid]

    ax3d.scatter(dots[0], dots[1], dots[2], facecolors=dot_colors, edgecolors=['white',] * n_cell,
                 alpha=0.7, s=8, clip_on=False, lw=0.5)
    ax2d.scatter(dots[0], dots[1], facecolors=dot_colors, edgecolors=['white',] * n_cell,
                 alpha=0.9, s=8, clip_on=False, lw=0.5)

    pos_dots, neg_dots = [], []
    for cell_cnt, cell_uid in enumerate(features.cells_uid):
        if cells_types[cell_uid] is CellType.Calb2_Pos:
            pos_dots.append((dots[0][cell_cnt], dots[1][cell_cnt]))
        elif cells_types[cell_uid] is CellType.Calb2_Neg:
            neg_dots.append((dots[0][cell_cnt], dots[1][cell_cnt]))
    pos_dots, neg_dots = np.array(pos_dots), np.array(neg_dots)
    # linear_reg(ax2d, pos_dots[:, 0], pos_dots[:, 1], lw=1, alpha=0.3, zorder=-2, ls='--',
    #            color=CELLTYPE2COLOR[CellType.Calb2_Pos])
    # linear_reg(ax2d, neg_dots[:, 0], neg_dots[:, 1], lw=1, alpha=0.3, zorder=-2, ls='--',
    #            color=CELLTYPE2COLOR[CellType.Calb2_Neg])

    ax2d.set_ylabel(f"{simplify_day_str(days1)} peak (log {DF_F0_STR})")
    ax2d.set_xlabel(f"{simplify_day_str(days2)} peak (log {DF_F0_STR})")
    ax3d.set_ylabel(f"{simplify_day_str(days1)} peak")
    ax3d.set_xlabel(f"{simplify_day_str(days2)} peak")

    yy, zz = np.meshgrid(np.linspace(-3, 1, 50), np.linspace(1, 3.5, 50))
    zticklabels = np.array([150, 200, 300, 900])

    ax3d.set_zlabel(f"Calb2 intensity (A.U.)")
    ax3d.view_init(elev=0, azim=-135)
    ax3d.set_box_aspect(aspect=(1, 1, 1))
    ax3d.set_zticks(CALB2_RESIZE_FUNC(zticklabels), zticklabels)
    ax2d.spines[['right', 'top']].set_visible(False)
    ax2d.plot(np.linspace(-4, 2, 50), np.linspace(-4, 2, 50),
              color='gray', alpha=0.6, lw=1)
    ax2d.set_aspect(1)
    for tmp_ax in (ax3d, ax2d,):
        tmp_ax.set_xlim(-3.5, 1.5)
        tmp_ax.set_ylim(-3.5, 1.5)
        tmp_ax.set_xticks([-3, -2, -1, 0, 1])
        tmp_ax.set_yticks([-3, -2, -1, 0, 1])
    fig2d.set_size_inches(2., 2.)
    fig3d.set_size_inches(3, 2)
    fig2d.tight_layout()
    fig3d.tight_layout()
    quick_save(fig2d, save_name+"_2d.png")
    quick_save(fig3d, save_name+"_3d.png")

