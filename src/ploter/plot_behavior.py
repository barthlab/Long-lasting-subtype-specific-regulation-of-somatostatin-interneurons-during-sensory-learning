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
    for single_trial in single_mice.trials:
        print(single_trial.elapsed_day, single_trial.lick_times[:5])
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


