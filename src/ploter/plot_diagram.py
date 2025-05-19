import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict, Counter

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.ploter.plotting_params import *
from src.ploter.plotting_utils import *
from src.ploter.statistic_annotation import *


def plot_diagram_large_view(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(1, 1)

    ax.plot(single_cs.fluorescence.t, single_cs.fluorescence.v, lw=1, alpha=0.8, color='black')
    # ax.spines[['bottom', 'left', 'right', 'top']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_size_inches(3, 1)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_trials(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(1, 1)

    for trial_id, single_trial in enumerate(single_cs.trials):  # type: int, Trial
        ax.plot(single_trial.df_f0.t, single_trial.df_f0.v, lw=1, alpha=0.8, color='black')
    # ax.spines[['bottom', 'left', 'right', 'top']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_size_inches(2, 0.7)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_spont_blocks(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(1, 1)

    for block_order, single_block in enumerate(single_cs.spont_blocks):  # type: int, SpontBlock
        ax.plot(single_block.df_f0.t, single_block.df_f0.v, lw=1, alpha=0.8, color='black')
    # ax.spines[['bottom', 'left', 'right', 'top']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_size_inches(2, 0.7)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_trials_colored(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(2, 1, sharex='all', gridspec_kw={'height_ratios': [6, 4]})

    puff_offset, blank_offset = 0, 0
    for trial_id, single_trial in enumerate(single_cs.trials):  # type: int, Trial
        if single_trial.trial_type is EventType.Puff:
            ax[0].plot(single_trial.df_f0.t_aligned, single_trial.df_f0.v - puff_offset,
                       lw=1, alpha=0.8, color=EVENT2COLOR[EventType.Puff])
            puff_offset += 1.
        else:
            ax[1].plot(single_trial.df_f0.t_aligned, single_trial.df_f0.v - blank_offset,
                       lw=1, alpha=0.8, color=EVENT2COLOR[EventType.Blank])
            blank_offset += 1.
    ax[0].axvspan(0, 0.5, color=OTHER_COLORS['puff'], alpha=0.3, lw=0)
    ax[1].axvline(x=0, lw=1, color=OTHER_COLORS['puff'], alpha=0.3,)
    ax[0].set_xticks([])
    for tmp_ax in ax:
        tmp_ax.set_yticks([])

    fig.set_size_inches(1, 1.5)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_spont_blocks_colored(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [0.5, 2, 0.5]})

    inter_offset = 0
    for block_order, single_block in enumerate(single_cs.spont_blocks):  # type: int, SpontBlock
        if single_block.block_type is BlockType.PreBlock:
            ax[0].plot(single_block.df_f0.t_zeroed, single_block.df_f0.v,
                       lw=1, alpha=0.8, color=EVENT2COLOR[BlockType.PreBlock])
        elif single_block.block_type is BlockType.PostBlock:
            ax[2].plot(single_block.df_f0.t_zeroed, single_block.df_f0.v,
                       lw=1, alpha=0.8, color=EVENT2COLOR[BlockType.PostBlock])
        else:
            ax[1].plot(single_block.df_f0.t_zeroed, single_block.df_f0.v + inter_offset,
                       lw=1, alpha=0.8, color=EVENT2COLOR[BlockType.InterBlock])
            inter_offset += 1.

    # ax.spines[['bottom', 'left', 'right', 'top']].set_visible(False)
    for tmp_ax in ax:
        tmp_ax.set_xticks([])
        tmp_ax.set_yticks([])

    fig.set_size_inches(1, 1.5)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_spont_blocks_event_detection(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(1, 1)

    example_block = general_filter(single_cs.spont_blocks, block_type=BlockType.PostBlock)[0]  # type: SpontBlock

    for std_cnt, std_value in enumerate([1, 2, 3, 5]):
        std_offset = std_cnt*10
        ax.plot(example_block.df_f0.t_zeroed, example_block.df_f0.v+std_offset, color='black', lw=1, alpha=0.7)
        ax.axhline(y=std_offset + std_value*single_cs.overall_baseline_std, xmin=0.2, lw=0.5, color='red', ls='--')
        peaks, properties = find_peaks(example_block.df_f0.v, prominence=std_value * single_cs.overall_baseline_std)
        for t_peak in peaks:
            ax.scatter(example_block.df_f0.t_zeroed[t_peak], example_block.df_f0.v[t_peak]+std_offset+2,
                       marker='v', color='red', alpha=0.7, s=5)
        ax.text(-5, std_offset, f"{std_value} " + r"$\sigma_2$", fontsize=8, ha='right', color='black')
    # ax.set_yticks(np.arange(len(OPTIONS_STD_RATIO)), [f"{std_option['ratio_std']} " + r"$\sigma_2$"
    #                                                   for std_option in OPTIONS_STD_RATIO.values()])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-28, None)
    # ax.spines[['bottom', 'left', 'right', 'top']].set_visible(False)

    fig.set_size_inches(1.5, 1.5)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_trial_periods(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(1, 1)

    example_df_f0, *_ = sync_timeseries([single_trial.df_f0 for single_trial in single_cs.trials])  # type: TimeSeries

    period_offset = 5
    for period_cnt, (period_name, period_option) in enumerate(OPTIONS_TIME_RANGE.items()):
        ax.plot(example_df_f0.t_aligned, example_df_f0.v+period_cnt*period_offset, lw=1, alpha=0.7, color='black')
        start_t, end_t = period_option["start_t"], period_option["end_t"]
        ax.add_patch(mpatches.Rectangle(
            (start_t, period_cnt*period_offset-0.2), end_t-start_t, 2.4,
            facecolor='gray', alpha=0.3, edgecolor='none', ))
        prefix = period_name[:-6]
        ax.text(-5, period_cnt*period_offset+1.7, f"{prefix}\nperiod", fontsize=8, ha='left', color='black')

    ax.set_xticks([-2, 0, 2, 4])
    ax.tick_params(axis="x", direction="in", pad=-15)
    ax.set_yticks([])
    ax.set_xlim(-5.5, None)
    ax.set_ylim(-2.7, 14)
    # ax.spines[['bottom', 'left', 'right', 'top']].set_visible(False)

    fig.set_size_inches(1.5, 2)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_prob_example(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(1, 1)
    trial_list = [single_cs.trials[trial_id] for trial_id in (0, 6, 7, 10, -2)]
    for std_cnt, std_value in enumerate([5, 10]):
        std_offset = std_cnt * 12
        for trial_id, single_trial in enumerate(trial_list):  # type: int, Trial
            if np.max(single_trial.df_f0.v) >= std_value*single_cs.trial_baseline_std:
                alpha, ls = 0.8, '-'
                ax.scatter(7*trial_id, np.max(single_trial.df_f0.v) + std_offset + 2,
                           marker='v', color='red', alpha=0.7, s=5)
            else:
                alpha, ls = 0.4, '--'
            ax.plot(single_trial.df_f0.t_aligned + 7*trial_id, single_trial.df_f0.v + std_offset,
                    lw=1, alpha=alpha, color='black', ls=ls)
            ax.axhline(y=std_offset + std_value * single_cs.trial_baseline_std, xmin=0.1, lw=0.5, color='red',
                       ls='--')
        ax.text(-5, std_offset, f"{std_value} " + r"$\sigma_1$", fontsize=8, ha='right', color='black')
    # ax.set_xticks([0, 7, 14, 21, 28], ["#1", "#2", "#3", "#4", "#5", ])
    ax.set_xticks([])
    ax.set_xlabel("Trials")
    ax.set_yticks([])
    ax.set_xlim(-6, None)
    ax.spines[['bottom', 'left', 'right', 'top']].set_visible(False)

    fig.set_size_inches(2., 1)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_trial_representative(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(1, 1)

    example_df_f0, *_ = sync_timeseries([single_trial.df_f0 for single_trial in single_cs.trials])  # type: TimeSeries

    ax.plot(example_df_f0.t_aligned, example_df_f0.v, lw=1, alpha=1, color='black')
    ax.axvline(x=0, lw=1, color='black', alpha=0.7, ls='--',)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top']].set_visible(False)

    fig.set_size_inches(1, 0.65)
    fig.tight_layout()
    quick_save(fig, save_name)


def plot_diagram_spont_representative(single_cs: CellSession, save_name: str):
    fig, ax = plt.subplots(1, 1)
    example_block = general_filter(single_cs.spont_blocks, block_type=BlockType.InterBlock)[1]  # type: SpontBlock

    ax.plot(example_block.df_f0.t_zeroed, example_block.df_f0.v, lw=1, alpha=1, color='black')
    ax.axhline(y=0.5 * single_cs.overall_baseline_std, lw=0.3, color='red', ls='--')
    peaks, properties = find_peaks(example_block.df_f0.v, prominence=0.5 * single_cs.overall_baseline_std)
    for t_peak in peaks:
        ax.scatter(example_block.df_f0.t_zeroed[t_peak], example_block.df_f0.v[t_peak] + 1,
                   marker='v', color='red', alpha=0.7, s=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top']].set_visible(False)

    fig.set_size_inches(1, 0.65)
    fig.tight_layout()
    quick_save(fig, save_name)


