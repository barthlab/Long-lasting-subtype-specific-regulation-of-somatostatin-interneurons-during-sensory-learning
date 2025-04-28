import numpy as np
import os
import os.path as path
from collections import defaultdict

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.feature.feature_manager import *
from src.ploter.plotting_params import *


def quick_save(fig, save_name):
    if FIGURE_SHOW_FLAG:
        plt.show()
    else:
        save_path = path.join(ROOT_PATH, FIGURE_PATH, save_name)
        os.makedirs(path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=900, transparent=True)
    plt.close(fig)


def oreo(ax: matplotlib.axes.Axes, list_timeseries: List[TimeSeries], mean_kwargs: dict, fill_kwargs: dict,
         x_offset: float = 0, y_offset: float = 0):
    avg_y, sem_y, (avg_xs, _) = sync_timeseries(list_timeseries)
    ax.plot(avg_xs + x_offset, avg_y.v + y_offset, **mean_kwargs)
    ax.fill_between(avg_xs + x_offset, avg_y.v - sem_y.v + y_offset, avg_y.v + sem_y.v + y_offset, **fill_kwargs)


def oreo_bar(ax: matplotlib.axes.Axes, list_values: List[float], x_position: float, width: float, **kwargs):
    mean_bar, sem_bar = nan_mean(list_values), nan_sem(list_values)
    ax.bar(x_position, mean_bar, yerr=sem_bar, error_kw=dict(lw=1, capsize=1, capthick=1), width=width, **kwargs)


def extract_avg_df_f0(single_image: Image, days_dict: Dict[str, Tuple[DayType, ...]], **trials_criteria) \
        -> Dict[str, Dict[CellUID, TimeSeries]]:
    extracted_data = defaultdict(dict)
    for group_name, group_of_days in days_dict.items():
        sub_image = single_image.select(day_id=group_of_days)
        sub_image_cell_split = sub_image.split("cell_uid")
        for cell_uid in single_image.cells_uid:
            all_trials = chain.from_iterable([single_cs.trials for single_cs in sub_image_cell_split[cell_uid].dataset])
            selected_trials = general_filter(all_trials, **trials_criteria)
            if len(selected_trials) == 0:
                raise ZeroDivisionError(f"Found zero available trials in {group_name} {cell_uid}")
            avg_df_f0, *_ = sync_timeseries([single_trial.df_f0 for single_trial in selected_trials])
            extracted_data[group_name][cell_uid] = avg_df_f0
    return extracted_data

