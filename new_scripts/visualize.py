import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
import numpy as np
from data_manager import *
from utils import *
from config import *
import os
import os.path as path


def plot_cell_session(single_cs: CellSession):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), width_ratios=[3, 1, 1])
    for stim_time, stim_label in zip(single_cs.stims.t, single_cs.stims.label):
        axs[0].axvspan(xmin=stim_time, xmax=stim_time + AP_DURATION,
                       color=EVENT2COLOR[stim_label], alpha=0.1)
    axs[0].set_ylim(-1, 3.0)
    axs[0].plot(single_cs.df_f0.t, single_cs.df_f0.v, lw=1, alpha=0.7, color='black')
    axs[0].fill_between(single_cs.df_f0.t, np.zeros_like(single_cs.df_f0.drop), single_cs.df_f0.drop,
                        lw=0, alpha=0.2, color='red')

    for trial_id, single_trial in enumerate(single_cs.trials):  # type: int, Trial
        axs[1].plot(single_trial.df_f0.t_aligned, single_trial.df_f0.v + trial_id * DY_DF_F0,
                    color='black', alpha=0.7, lw=1)
        axs[1].add_patch(ptchs.Rectangle(
            (0, (trial_id - 0.4) * DY_DF_F0), 0.5, 0.8*DY_DF_F0,
            facecolor=EVENT2COLOR[single_trial.trial_type], alpha=0.3, edgecolor='none', ))

    for block_order, single_block in enumerate(single_cs.spont_blocks):
        print(single_block)
        axs[2].plot(single_block.df_f0.t_aligned, single_block.df_f0.v + (block_order-0.5) * DY_DF_F0,
                    color=EVENT2COLOR[single_block.block_type], alpha=0.7, lw=1)
        axs[0].add_patch(ptchs.Rectangle(
            (single_block.block_start, -1.5 * DY_DF_F0), single_block.block_len, DY_DF_F0,
            facecolor=EVENT2COLOR[single_block.block_type], alpha=0.3, edgecolor='none', ))
    axs[2].sharey(axs[1])
    plt.show()