import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
from collections import defaultdict
from itertools import chain

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.feature.feature_manager import *
from src.ploter.plotting_params import *


def plot_heatmap_overview_cellwise(
        single_image: Image, feature_db: FeatureDataBase, save_name: str,
        cols_ref: Dict[str, Tuple[DayType, ...]], cols_names: List[str],
        trials_criteria: dict = None, sorting: Tuple[str, Callable[TimeSeries, float]] = None, theme_color: str = "black"
):
    n_col = len(cols_names)
    trials_criteria = trials_criteria if trials_criteria is not None else {}

    # extract data
    extracted_data = defaultdict(dict)
    for col_name in cols_names:
        days_in_col = cols_ref[col_name]
        sub_image = single_image.select(day_id=days_in_col)
        sub_image_cell_split = sub_image.split("cell_uid")
        for cell_uid in single_image.cells_uid:
            all_trials = chain.from_iterable([single_cs.trials for single_cs in sub_image_cell_split[cell_uid].dataset])
            selected_trials = general_filter(all_trials, **trials_criteria)
            if len(selected_trials) == 0:
                raise ZeroDivisionError(f"Found zero available trials")
            avg_df_f0, *_ = sync_timeseries([single_trial.df_f0 for single_trial in selected_trials])
            extracted_data[col_name][cell_uid] = avg_df_f0

    # sorting
    sort_col = "Record Date"
    if sorting is not None:
        sort_col, sort_func = sorting
        assert sort_col in cols_names, f"Sorting name missing ({sort_col}) in {cols_names}"
        cells_uid_order = sorted(single_image.cells_uid, key=lambda x: sort_func(extracted_data[sort_col][x]),
                                 reverse=True)
    else:
        cells_uid_order = single_image.cells_uid

    # plotting
    fig, axs = plt.subplots(2, n_col+1, sharex="col", width_ratios=[1] * n_col + [0.08], height_ratios=[1, 0.2])
    plt.subplots_adjust(wspace=0.15)
    axh, axc = axs[0, :], axs[1, :]
    for ax_id, col_name in enumerate(cols_names):
        grand_avg_df_f0, grand_sem_df_f0, (xs, grand_matrix) = sync_timeseries(
            [extracted_data[col_name][cell_uid] for cell_uid in cells_uid_order])

        im = axh[ax_id].imshow(
            grand_matrix,
            aspect='auto',
            origin='upper',
            vmin=DISPLAY_MIN_DF_F0, vmax=DISPLAY_MAX_DF_F0,
            cmap='viridis',
            interpolation='nearest'
        )
        axc[ax_id].plot(grand_avg_df_f0.v, lw=LW_SMALL_DF_F0, color=theme_color, alpha=ALPHA_DEEP_DF_F0)
        axc[ax_id].fill_between(np.arange(len(xs)), grand_avg_df_f0.v - grand_sem_df_f0.v,
                                grand_avg_df_f0.v + grand_sem_df_f0.v, lw=0, color=theme_color,
                                alpha=ALPHA_LIGHT_DF_F0)

        axc[ax_id].spines[['right', 'top']].set_visible(False)
        if ax_id == 0:
            axh[ax_id].set_ylabel(f"Cell ID (Ranked on {sort_col})" if sorting is not None else "Cell ID")
            axc[ax_id].set_ylabel(r'$\Delta F/F_0$')
        else:
            axh[ax_id].set_yticklabels([])
            axh[ax_id].set_yticks([])
            axc[ax_id].set_yticklabels([])
            axc[ax_id].set_yticks([])

        x_tick_loc, x_tick_pos = [-1, 0, 1, 2], []
        for x_tick in x_tick_loc:
            x_tick_pos.append(np.searchsorted(xs, x_tick))
        axc[ax_id].set_xlim(np.searchsorted(xs, -2), np.searchsorted(xs, 3))
        axh[ax_id].axvline(x=np.searchsorted(xs, 0), color='red', alpha=0.7, ls='--', lw=0.5)
        axc[ax_id].axvspan(np.searchsorted(xs, 0), np.searchsorted(xs, 0.5), lw=0, color='green', alpha=0.4)
        axh[ax_id].set_xticks(x_tick_pos, x_tick_loc)
        axc[ax_id].set_xticks(x_tick_pos, x_tick_loc)
        axh[ax_id].set_xlabel("Time [s]")
        axc[ax_id].set_xlabel("Time [s]")
        axh[ax_id].set_title(col_name)

    cbar = fig.colorbar(im, cax=axh[-1])
    cbar.set_label(r'$\Delta F/F_0$')
    axc[-1].remove()

    fig.set_size_inches(7.5, 3)
    fig.tight_layout()
    if FIGURE_SHOW_FLAG:
        plt.show()
    else:
        save_path = path.join(ROOT_PATH, FIGURE_PATH, save_name)
        os.makedirs(path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)
