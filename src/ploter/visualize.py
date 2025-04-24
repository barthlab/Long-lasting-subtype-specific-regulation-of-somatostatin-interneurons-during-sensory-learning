import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
import matplotlib
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.interpolate import splprep, splev

import numpy as np
import os
import os.path as path

from src.data_manager import *
from src.basic.utils import *
from src.config import *
from src.feature.feature_manager import *

plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 8


def oreo(ax: matplotlib.axes.Axes, trials: List[Trial], mean_kwargs: dict, fill_kwargs: dict,
         x_offset: float = 0, y_offset: float = 0):
    avg_xs, avg_mean, avg_sem, _ = synchronize_time_series_data(
        [single_trial.df_f0.t_aligned for single_trial in trials],
        [single_trial.df_f0.v for single_trial in trials],
    )
    ax.plot(avg_xs + x_offset, avg_mean + y_offset, **mean_kwargs)
    ax.fill_between(avg_xs + x_offset, avg_mean - avg_sem + y_offset, avg_mean + avg_sem + y_offset, **fill_kwargs)


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
        axs[1].add_patch(ptchs.Rectangle(
            (0, (trial_id - 0.4) * DY_DF_F0), AP_DURATION, 0.8*DY_DF_F0,
            facecolor=EVENT2COLOR[single_trial.trial_type], alpha=0.3, edgecolor='none', ))

    for block_order, single_block in enumerate(single_cs.spont_blocks):
        axs[2].plot(single_block.df_f0.t_aligned, single_block.df_f0.v + (block_order-0.5) * DY_DF_F0,
                    color=EVENT2COLOR[single_block.block_type], alpha=0.7, lw=1)
        axs[0].add_patch(ptchs.Rectangle(
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
    if FIGURE_SHOW_FLAG:
        plt.show()
    else:
        fig.savefig(path.join(ROOT_PATH, FIGURE_PATH, save_name), bbox_inches='tight', dpi=500)
    plt.close(fig)


def plot_image(single_image: Image, save_name: str, title: str = None,
               additional_info: FeatureDataBase = None):
    num_cell, num_days = len(single_image.cells_uid), len(single_image.days)
    fig, axs = plt.subplots(num_days + 1, 3 * num_cell, width_ratios=[1, 1, 2]*num_cell,
                            sharey='row', sharex='col')
    print(f"Cell num: {len(single_image.cells_uid)}", [x for x in single_image.cells_uid])
    print([x.name for x in single_image.days])
    cell_types = {CellType.Unknown for _ in single_image.cells_uid} if additional_info is None else (
        additional_info.cell_types)

    # plot individual data
    max_depth = 0
    cell_image_dict = single_image.cell_split()
    for col_id, (cell_uid, cell_image) in enumerate(cell_image_dict.items()):  # type: int, (CellUID, Image)
        day_image_dict_percell = cell_image.day_split()
        postfix = f" {additional_info.get('Calb2 Mean').get_cell(cell_uid):.1f}" if additional_info is not None\
            else ""
        axs[0, col_id * 3 + 1].set_title(cell_uid.in_short()+postfix, color=CELLTYPE2COLOR[cell_types[cell_uid]])

        for row_id, (single_day, cell_day_image) in enumerate(day_image_dict_percell.items()):  # type: int, (SatDay|PseDay, Image)
            print(row_id, col_id, cell_day_image.cells_uid, cell_day_image.days, len(cell_day_image.dataset))
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


                    evoked_period = single_trial.df_f0.segment(0, TEST_EVOKED_PERIOD, relative_flag=True).v
                    baseline_period = single_trial.df_f0.segment(*TRIAL_BASELINE_RANGE, relative_flag=True).v
                    if np.mean(evoked_period) >= TEST_STD_RATIO*np.std(baseline_period):
                        tmp_ax.scatter(-2, - trial_id * DY_DF_F0 - session_dy, marker="*", color="black", s=10)


                for block_order, single_block in enumerate(single_cs.spont_blocks):
                    spont_ax.plot(single_block.df_f0.t_aligned,
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

        oreo(ax=avg_puff_ax, trials=general_filter(all_trials, trial_type=EventType.Puff),
             mean_kwargs={"color": EVENT2COLOR[EventType.Puff], "alpha": 0.7, "lw": 1},
             fill_kwargs={"color": EVENT2COLOR[EventType.Puff], "alpha": 0.2, "lw": 0})
        avg_puff_ax.axvspan(0, AP_DURATION, color=OTHER_COLORS["puff"], alpha=0.1, lw=0)

        oreo(ax=avg_blank_ax, trials=general_filter(all_trials, trial_type=EventType.Blank),
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
            axs[row_id, col_id].set_ylim(max_depth, DISPLAY_MAX_DF_F0)

    if title is not None:
        fig.suptitle(title)
    fig.set_size_inches(num_cell*8, 1.5*(num_days+1))
    fig.tight_layout()
    if FIGURE_SHOW_FLAG:
        plt.show()
    else:
        save_path = path.join(ROOT_PATH, FIGURE_PATH, save_name)
        os.makedirs(path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_plasticity_manifold(features: FeatureDataBase, save_name: str):
    features.compute_DayWiseFeature("EvokedPeak", compute_evoked_peak)
    features.compute_DayWiseFeature("ResponseProb", compute_response_prob)

    for period_name, period_list in zip(
        ["ACC456", "SAT123", "SAT456",], [[3, 4, 5], [6, 7, 8], [9, 10, 11]]
    ):
        for target_feature_name in ("EvokedPeak", "ResponseProb",):
            features.compute_CellWiseFeature(
                f"{target_feature_name}_{period_name}",
                lambda cell_uid, features: daywise_average(
                    cell_uid, features, target_feature_name, period_list))

    evoked_peak, response_prob = features.get("EvokedPeak"), features.get("ResponseProb")
    evoked_peak_acc456, response_prob_acc456 = features.get("EvokedPeak_ACC456"), features.get("ResponseProb_ACC456")
    evoked_peak_sat123, response_prob_sat123 = features.get("EvokedPeak_SAT123"), features.get("ResponseProb_SAT123")
    evoked_peak_sat456, response_prob_sat456 = features.get("EvokedPeak_SAT456"), features.get("ResponseProb_SAT456")
    calb2 = features.get("Calb2")
    calb2_mean = features.get("Calb2 Mean")

    fig = plt.figure()
    ax_3d1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax_3d2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax_2d = fig.add_subplot(1, 3, 3)
    for cell_cnt, cell_uid in enumerate(features.cells_uid):
        evoked_peak_s = evoked_peak_sat123.get_cell(cell_uid)
        response_prob_s = response_prob_sat123.get_cell(cell_uid)
        evoked_peak_e = evoked_peak_sat456.get_cell(cell_uid)
        response_prob_e = response_prob_sat456.get_cell(cell_uid)
        cell_calb2_intensity = calb2_mean.get_cell(cell_uid)

        s_point = np.array((np.log(0.1 + evoked_peak_s), 100*response_prob_s, np.log(cell_calb2_intensity-60)))
        e_point = np.array((np.log(0.1 + evoked_peak_e), 100*response_prob_e, np.log(cell_calb2_intensity-60)))

        num_segments = 20
        t = np.linspace(0, 1, num_segments)
        cmap = cm.get_cmap("Reds") if calb2.get_cell(cell_uid) == 1 else cm.get_cmap("Greys")
        colors = [cmap(i / num_segments) for i in range(num_segments)]
        points = np.array([s_point + (e_point - s_point) * i for i in t])

        segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
        lc = Line3DCollection(segments, colors=colors, linewidths=2, alpha=0.5)
        ax_3d1.add_collection3d(lc)
        lc = Line3DCollection(segments, colors=colors, linewidths=2, alpha=0.5)
        ax_3d2.add_collection3d(lc)

        segments = [(points[i, :2], points[i + 1, :2]) for i in range(len(points) - 1)]
        lc = LineCollection(segments, colors=colors, linewidths=2, alpha=0.5)
        ax_2d.add_collection(lc)

        ax_3d1.scatter(*e_point, color=CELLTYPE2COLOR[features.cell_types[cell_uid]], alpha=0.7, s=10)
        ax_3d2.scatter(*e_point, color=CELLTYPE2COLOR[features.cell_types[cell_uid]], alpha=0.7, s=10)
        ax_2d.scatter(*e_point[:2], color=CELLTYPE2COLOR[features.cell_types[cell_uid]], alpha=0.7, s=10)

        # ax_3d1.scatter(*s_point, color=CELLTYPE2COLOR[features.cell_types[cell_uid]], alpha=0.7, s=10)
        # ax_3d2.scatter(*s_point, color=CELLTYPE2COLOR[features.cell_types[cell_uid]], alpha=0.7, s=10)
        # ax_2d.scatter(*s_point[:2], color=CELLTYPE2COLOR[features.cell_types[cell_uid]], alpha=0.7, s=10)

    x_coords = [0.1, 0.5, 1, 1.5]
    y_coords = [0, 20, 40, 60, 80, 100]
    z_coords = [100, 200, 300, 400, 500]
    for tmp_ax in (ax_3d1, ax_3d2, ax_2d):
        tmp_ax.autoscale_view()
        tmp_ax.set_xlim(-3, 1)
        tmp_ax.set_ylim(-10, 110)
        tmp_ax.set_xticks(np.log(0.1 + np.array(x_coords)), x_coords)
        tmp_ax.set_yticks(y_coords)
        tmp_ax.set_xlabel(r"Evoked Response [$\Delta F/F_0$]")
        tmp_ax.set_ylabel("Response Probability [%]")
    for tmp_ax in (ax_3d1, ax_3d2,):
        tmp_ax.set_zlim(4.5, 8.0)
        tmp_ax.set_zticks(np.log(np.array(z_coords)-60), z_coords)
        tmp_ax.set_zlabel("mCherry Fluorescence [A.U.]")
    ax_2d.spines[['right', 'top']].set_visible(False)
    ax_3d1.view_init(elev=30, azim=-45)
    ax_3d2.view_init(elev=0, azim=-45)


    fig.subplots_adjust(wspace=0.2)

    fig.set_size_inches(18, 4)
    fig.tight_layout()

    if FIGURE_SHOW_FLAG:
        plt.show()
    else:
        save_path = path.join(ROOT_PATH, FIGURE_PATH, save_name)
        os.makedirs(path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)