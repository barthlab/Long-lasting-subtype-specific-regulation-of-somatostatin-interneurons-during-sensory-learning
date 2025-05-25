import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict
from itertools import chain
import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import viridis, Greys9

from src.basic.utils import *
from src.config import *
from src.feature.feature_manager import *
from src.ploter.plotting_params import *
from src.ploter.plotting_utils import *
from src.ploter.statistic_annotation import *
from src.behavior.behavior_manager import *


def plot_heatmap_licks(single_mice: BehaviorMice, save_name: str):
    lick_times_list = []
    elapsed_days_list = []
    trial_types_list = []

    for single_trial in single_mice.trials:
        trial_type_str = 'Go' if single_trial.trial_type is BehaviorTrialType.Go else 'NoGo'
        for lick_time in single_trial.lick_times:
            if BEHAVIOR_RANGE[0] <= lick_time <= BEHAVIOR_RANGE[1]:
                lick_times_list.append(lick_time)
                elapsed_days_list.append(single_trial.elapsed_day)
                trial_types_list.append(trial_type_str)

    df = pd.DataFrame({
        'lick_time': lick_times_list,
        'elapsed_day': elapsed_days_list,
        'trial_type': trial_types_list
    })

    x_range = (df['lick_time'].min(), df['lick_time'].max())
    y_range = (df['elapsed_day'].min(), df['elapsed_day'].max())

    cvs = ds.Canvas(plot_width=int((x_range[1]-x_range[0])/D_TIME),
                    plot_height=int((y_range[1]-y_range[0])/D_DAY))
    agg_go = cvs.points(df[df['trial_type'] == 'Go'], 'lick_time', 'elapsed_day', agg=ds.count())
    agg_nogo = cvs.points(df[df['trial_type'] == 'NoGo'], 'lick_time', 'elapsed_day', agg=ds.count())
    binary_agg_go = (agg_go > 0).astype(int)
    binary_agg_nogo = (agg_nogo > 0).astype(int)
    img_go = tf.shade(binary_agg_go, cmap=BINARY)
    img_nogo = tf.shade(binary_agg_nogo, cmap=BINARY)

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

    fig, axs = plt.subplots(1, 2, sharex='all')

    axs[0].imshow(img_go.to_pil(), extent=extent, origin='upper', aspect='auto', interpolation='nearest')
    axs[1].imshow(img_nogo.to_pil(), extent=extent, origin='upper', aspect='auto', interpolation='nearest')
    axs[0].set_xlim(*BEHAVIOR_RANGE)
    axs[0].set_xticks([0, 1, 3])
    axs[0].set_ylim(17, 0)
    axs[1].set_ylim(17, 0)

    for i, day_id in enumerate(EXP2DAY[single_mice.exp_id]):
        axs[0].text(-0.05, i + 1, day_id.name, transform=axs[0].get_yaxis_transform(), fontsize=DAY_TEXT_SIZE,
                    ha='right', va='center', clip_on=False)
        axs[1].text(-0.05, i + 1, day_id.name, transform=axs[1].get_yaxis_transform(), fontsize=DAY_TEXT_SIZE,
                    ha='right', va='center', clip_on=False)
    axs[0].set_yticks([0.5 + i for i in range(17)], ["" for _ in range(17)])
    axs[1].set_yticks([0.5 + i for i in range(17)], ["" for _ in range(17)])
    axs[0].axvspan(0, 0.5, lw=0, color=OTHER_COLORS['puff'], alpha=0.1, zorder=0)

    axs[0].set_title('Go Trials')
    axs[1].set_title('NoGo Trials')
    fig.suptitle(single_mice.mice_id)
    axs[0].set_xlabel('Lick Time (s)')
    axs[1].set_xlabel('Lick Time (s)')
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[1].spines[['right', 'top']].set_visible(False)

    fig.set_size_inches(3, 5)

    # plt.tight_layout()  # Adjust layout
    plt.show()
    plt.close(fig)


def plot_single_day_performance(ax: matplotlib.pyplot.Axes, daily_trials: List[BehaviorTrial],):

    for trial_type, scale_factor in zip(
            [BehaviorTrialType.Go, BehaviorTrialType.NoGo],
            [1, -1],
    ):
        daily_specific_trials = general_filter(daily_trials, trial_type=trial_type)
        trial_times = [single_trial.daily_hour for single_trial in daily_specific_trials]
        trial_lick_freq = [scale_factor * single_trial.anticipatory_licking for single_trial in daily_specific_trials]

        bin_time, bin_lick_freq, bin_var = bin_average(trial_times, trial_lick_freq,
                                                       bin_width=BEHAVIOR_BIN_SIZE_HOUR)

        # tmp_ax.scatter(trial_times, trial_lick_freq, alpha=0.3, s=5,
        #             color=BEHAVIOR_TRIAL_TYPE2COLOR[trial_type])
        ax.bar(bin_time, bin_lick_freq, width=BEHAVIOR_BIN_SIZE_HOUR * 0.25, alpha=0.8,
               yerr=bin_var, capsize=1, error_kw={"capthick": 0.5, "elinewidth": 0.5,},
               color=BEHAVIOR_TRIAL_TYPE2COLOR[trial_type])
    ax.plot([0, 24], [0, 0], lw=1, color='gray', alpha=0.8)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both')


def plot_single_day_licking_raster(ax: matplotlib.pyplot.Axes, daily_trials: List[BehaviorTrial], color: str):
    n_trials = len(daily_trials)
    lick_times = []
    for single_trial in daily_trials:
        lick_times += list(single_trial.lick_times)

    bin_time, bin_lick_freq = bin_count(lick_times, bin_width=BEHAVIOR_BIN_SIZE_TRIAL)
    bin_lick_freq = bin_lick_freq/(n_trials*BEHAVIOR_BIN_SIZE_TRIAL)
    ax.plot(bin_time, bin_lick_freq, lw=1, alpha=0.8, color=color)

    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both')
    ax.set_ylim(0, 8)


def plot_daily_performance(single_mice: BehaviorMice, save_name: str):
    trials_by_days = single_mice.split_trials_by_days()
    n_day = len(trials_by_days.keys())
    n_row, n_col = row_col_from_n_subplots(n_day)

    fig, axs = plt.subplots(n_row, n_col, sharey='all', sharex='all')

    for day_id, day_name in enumerate(trials_by_days.keys()):
        tmp_ax = axs[int(np.floor(day_id/n_col)), day_id % n_col]
        daily_trials = trials_by_days[day_name]
        plot_single_day_performance(tmp_ax, daily_trials)
        tmp_ax.set_xlabel(f"Day{day_name} (Hour)")

    fig.set_size_inches(2*n_col, 2*n_row)
    fig.tight_layout()
    # quick_save(fig, save_name)
    plt.show()


def plot_daily_summary(
        beh_exp: BehaviorExperiment, img_exp: Experiment,
        img_groups: List[dict], color_groups: List[str], col_days: Dict[str, List[DayType]], save_name: str
):
    n_group, n_col = len(img_groups), len(col_days[list(col_days.keys())[0]])

    fig, axs = plt.subplots(n_group+2, n_col, sharey='row', height_ratios=[1, ] * n_group + [0.8, 1])

    for col_id in range(n_col):
        for row_id, select_criteria in enumerate(img_groups):

            all_trials = chain.from_iterable([
                single_cs.trials for beh_mice in beh_exp.mice for single_cs in general_filter(
                    img_exp.get_mice(beh_mice.mice_uid).cell_sessions,
                    day_id=col_days[beh_mice.mice_id][col_id], **select_criteria)])
            selected_trials = general_filter(all_trials, trial_type=EventType.Puff)
            if len(selected_trials) == 0:
                continue
            tmp_ax = axs[row_id, col_id]
            oreo(
                tmp_ax, [single_trial.df_f0 for single_trial in selected_trials],
                mean_kwargs={"alpha": 0.7, "lw": 1, "color": color_groups[row_id]},
                fill_kwargs={"alpha": 0.2, "lw": 0, "color": color_groups[row_id]},
            )

            tmp_ax.spines[['right', 'top']].set_visible(False)
            tmp_ax.tick_params(axis='both')
            if col_id == 0:
                tmp_ax.set_ylabel(f"Evoked Response ({DF_F0_STR})")
            tmp_ax.set_xlabel("Time (s)")
            tmp_ax.set_xlim(-2, 4)
            tmp_ax.axvspan(0, 0.5, lw=0, color=OTHER_COLORS['puff'], alpha=0.4)

        daily_trials = list(chain.from_iterable([beh_mice.split_trials_by_days().get(
            col_days[beh_mice.mice_id][col_id].value+1, []) for beh_mice in beh_exp.mice]))
        plot_single_day_performance(axs[n_group+1, col_id], daily_trials)
        if col_id == 0:
            axs[n_group+1, col_id].set_ylabel(f"Anticipatory Licking Freq (Hz)")
        axs[n_group+1, col_id].set_xlabel(f"Hour at day")

        plot_single_day_licking_raster(axs[n_group, col_id],
                                       general_filter(daily_trials, trial_type=BehaviorTrialType.Go),
                                       color=BEHAVIOR_TRIAL_TYPE2COLOR[BehaviorTrialType.Go])
        plot_single_day_licking_raster(axs[n_group, col_id],
                                       general_filter(daily_trials, trial_type=BehaviorTrialType.NoGo),
                                       color=BEHAVIOR_TRIAL_TYPE2COLOR[BehaviorTrialType.NoGo])
        if col_id == 0:
            axs[n_group, col_id].set_ylabel(f"Licking Freq (Hz)")
        axs[n_group, col_id].set_xlim(-2, 4)
        axs[n_group, col_id].axvspan(0, 0.5, lw=0, color=OTHER_COLORS['puff'], alpha=0.4)
        axs[n_group, col_id].axvline(x=1, lw=1, color=OTHER_COLORS['water'], alpha=0.4, ls='--')

    fig.set_size_inches(1.3*n_col, 0.9*(n_group+3))
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_daily_bar_graph(
        beh_exp: BehaviorExperiment, img_exp: Experiment,
        img_groups: List[dict], color_groups: List[str], col_days: Dict[str, List[DayType]], save_name: str
):
    n_group, n_days = len(img_groups), len(col_days[list(col_days.keys())[0]])

    fig, axs = plt.subplots(n_group+1, 1)

    cell_peaks = {i: {} for i in range(n_group)}
    beh_trial_list = {}
    anti_licks_pairs = {}

    for col_id in range(n_days):
        for row_id, select_criteria in enumerate(img_groups):
            cell_peaks[row_id][col_id] = defaultdict(list)
            all_trials = chain.from_iterable([
                single_cs.trials for beh_mice in beh_exp.mice for single_cs in general_filter(
                    img_exp.get_mice(beh_mice.mice_uid).cell_sessions,
                    day_id=col_days[beh_mice.mice_id][col_id], **select_criteria)])
            selected_trials = general_filter(all_trials, trial_type=EventType.Puff)
            if len(selected_trials) == 0:
                continue

            for single_trial in selected_trials:
                cell_peaks[row_id][col_id][single_trial.cell_uid].append(
                    element_feature_compute_trial_period_activity_peak(single_trial, 0, EVOKED_PERIOD)
                )
            for single_cell_uid in cell_peaks[row_id][col_id].keys():
                cell_peaks[row_id][col_id][single_cell_uid] = nan_mean(cell_peaks[row_id][col_id][single_cell_uid])

        beh_trial_list[col_id] = defaultdict(list)
        anti_licks_pairs[col_id] = {}
        daily_trials = list(chain.from_iterable([beh_mice.split_trials_by_days().get(
            col_days[beh_mice.mice_id][col_id].value+1, []) for beh_mice in beh_exp.mice]))

        for single_trial in daily_trials:
            beh_trial_list[col_id][single_trial.mice_uid].append(single_trial)
        for mice_uid in beh_trial_list[col_id].keys():
            daily_single_mice_trials = beh_trial_list[col_id][mice_uid]
            anti_licks_pairs[col_id][mice_uid] = (
                calculate_last_percent_anticipatory_licking(
                    general_filter(daily_single_mice_trials, trial_type=BehaviorTrialType.Go)),
                calculate_last_percent_anticipatory_licking(
                    general_filter(daily_single_mice_trials, trial_type=BehaviorTrialType.NoGo)),
            )

    """
    TODO: Simplify the following Code
    """

    # Plot cell peaks for each group (axs[0] to axs[n_group-1])
    for group_id in range(n_group):
        ax = axs[group_id]

        paired_ttest_with_Bonferroni_correction(
            ax, {
                3: cell_peaks[group_id][3],
                0: cell_peaks[group_id][0],
                1: cell_peaks[group_id][1],
                2: cell_peaks[group_id][2],
                4: cell_peaks[group_id][4],
                5: cell_peaks[group_id][5],
            }, simple_flag=True)
        # Collect all unique cell UIDs across all days for this group
        all_cell_uids = set()
        for col_id in range(n_days):
            if col_id in cell_peaks[group_id]:
                all_cell_uids.update(cell_peaks[group_id][col_id].keys())

        # Prepare data for plotting
        x_values = list(range(n_days))
        y_values = []
        y_errors = []

        # Individual cell trajectories for connecting lines
        cell_trajectories = {cell_uid: [] for cell_uid in all_cell_uids}

        for col_id in range(n_days):
            if col_id in cell_peaks[group_id]:
                daily_values = list(cell_peaks[group_id][col_id].values())
                y_values.append(np.nanmean(daily_values) if daily_values else np.nan)
                y_errors.append(np.nanstd(daily_values) / np.sqrt(len(daily_values)) if daily_values else 0)

                # Store individual cell values for trajectories
                for cell_uid in all_cell_uids:
                    if cell_uid in cell_peaks[group_id][col_id]:
                        cell_trajectories[cell_uid].append((col_id, cell_peaks[group_id][col_id][cell_uid]))
                    else:
                        cell_trajectories[cell_uid].append((col_id, np.nan))
            else:
                y_values.append(np.nan)
                y_errors.append(0)
                for cell_uid in all_cell_uids:
                    cell_trajectories[cell_uid].append((col_id, np.nan))

        # Plot error bars
        ax.errorbar(x_values, y_values, yerr=y_errors, fmt='o-',
                    color=color_groups[group_id], lw=2, markersize=4, capsize=4)

        # Plot individual cell trajectories
        for cell_uid, trajectory in cell_trajectories.items():
            x_traj = [point[0] for point in trajectory if not np.isnan(point[1])]
            y_traj = [point[1] for point in trajectory if not np.isnan(point[1])]
            if len(x_traj) > 1:  # Only plot if there are at least 2 points
                # ax.plot(x_traj, y_traj, alpha=0.1, linewidth=0.5, color=color_groups[group_id], ls='--')
                pass
        # ax.set_xlabel('Day relative to')
        ax.set_ylabel(f'Evoked Response (log {DF_F0_STR})')
        # ax.set_yscale('log')
        ax.spines[['right', 'top']].set_visible(False)

        ax.set_xticks(x_values, ["ACC6", -2, -1, 0, 1, 2])
        # ax.grid(True, alpha=0.3)

    # Plot behavioral data (axs[n_group])
    ax = axs[n_group]

    # Collect all unique mice UIDs across all days
    all_mice_uids = set()
    for col_id in range(n_days):
        if col_id in anti_licks_pairs:
            all_mice_uids.update(anti_licks_pairs[col_id].keys())

    # Prepare data for Go and NoGo trials
    go_values = []
    go_errors = []
    nogo_values = []
    nogo_errors = []

    # Individual mice trajectories
    mice_go_trajectories = {mice_uid: [] for mice_uid in all_mice_uids}
    mice_nogo_trajectories = {mice_uid: [] for mice_uid in all_mice_uids}

    for col_id in range(n_days):
        if col_id in anti_licks_pairs:
            go_daily = [anti_licks_pairs[col_id][mice_uid][0] for mice_uid in anti_licks_pairs[col_id].keys()]
            nogo_daily = [anti_licks_pairs[col_id][mice_uid][1] for mice_uid in anti_licks_pairs[col_id].keys()]

            # paired_ttest_with_Bonferroni_correction(
            #     ax, {
            #         col_id: {mice_uid: anti_licks_pairs[col_id][mice_uid][0] for mice_uid in anti_licks_pairs[col_id].keys()},
            #         col_id+0.25: {mice_uid: anti_licks_pairs[col_id][mice_uid][1] for mice_uid in anti_licks_pairs[col_id].keys()},
            #     }, simple_flag=False)

            go_values.append(np.nanmean(go_daily) if go_daily else np.nan)
            go_errors.append(np.nanstd(go_daily) / np.sqrt(len(go_daily)) if go_daily else 0)
            nogo_values.append(np.nanmean(nogo_daily) if nogo_daily else np.nan)
            nogo_errors.append(np.nanstd(nogo_daily) / np.sqrt(len(nogo_daily)) if nogo_daily else 0)

            # Store individual mice values for trajectories
            for mice_uid in all_mice_uids:
                if mice_uid in anti_licks_pairs[col_id]:
                    mice_go_trajectories[mice_uid].append((col_id, anti_licks_pairs[col_id][mice_uid][0]))
                    mice_nogo_trajectories[mice_uid].append((col_id + 0.25, anti_licks_pairs[col_id][mice_uid][1]))
                else:
                    mice_go_trajectories[mice_uid].append((col_id, np.nan))
                    mice_nogo_trajectories[mice_uid].append((col_id + 0.25, np.nan))
        else:
            go_values.append(np.nan)
            go_errors.append(0)
            nogo_values.append(np.nan)
            nogo_errors.append(0)
            for mice_uid in all_mice_uids:
                mice_go_trajectories[mice_uid].append((col_id, np.nan))
                mice_nogo_trajectories[mice_uid].append((col_id + 0.25, np.nan))

    # Plot Go trials (at col_id)
    x_go = list(range(n_days))
    ax.errorbar(x_go, go_values, yerr=go_errors, fmt='o-',
                color=BEHAVIOR_TRIAL_TYPE2COLOR[BehaviorTrialType.Go],
                linewidth=2, markersize=4, capsize=5, label='Go')

    # Plot NoGo trials (at col_id + 0.25)
    x_nogo = [x + 0.25 for x in range(n_days)]
    ax.errorbar(x_nogo, nogo_values, yerr=nogo_errors, fmt='s-',
                color=BEHAVIOR_TRIAL_TYPE2COLOR[BehaviorTrialType.NoGo],
                linewidth=2, markersize=4, capsize=5, label='NoGo')

    # Plot individual mice trajectories
    for mice_uid in all_mice_uids:
        # Go trajectories
        go_traj = mice_go_trajectories[mice_uid]
        x_go_traj = [point[0] for point in go_traj if not np.isnan(point[1])]
        y_go_traj = [point[1] for point in go_traj if not np.isnan(point[1])]
        if len(x_go_traj) > 1:
            ax.plot(x_go_traj, y_go_traj, alpha=0.3, linewidth=0.5,
                    color=BEHAVIOR_TRIAL_TYPE2COLOR[BehaviorTrialType.Go], ls='--')

        # NoGo trajectories
        nogo_traj = mice_nogo_trajectories[mice_uid]
        x_nogo_traj = [point[0] for point in nogo_traj if not np.isnan(point[1])]
        y_nogo_traj = [point[1] for point in nogo_traj if not np.isnan(point[1])]
        if len(x_nogo_traj) > 1:
            ax.plot(x_nogo_traj, y_nogo_traj, alpha=0.3, linewidth=0.5,
                    color=BEHAVIOR_TRIAL_TYPE2COLOR[BehaviorTrialType.NoGo], ls='--')

    # ax.set_xlabel('Day')
    ax.set_ylabel('Anticipatory Licking Freq (Hz)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks(list(range(n_days)), ["ACC6", -2, -1, 0, 1, 2])
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    fig.set_size_inches(0.5*n_days, 1.5*(n_group+1))
    fig.tight_layout()
    quick_save(fig, save_name)
