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
import seaborn as sns


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

#
# def pool_boolean_array(arr: np.ndarray, xs: np.ndarray, threshold=0.5) -> np.ndarray:
#     tmp_fs = (len(xs)-1) / (xs[-1]-xs[0])
#     window_size = int(SIGNIFICANT_TRACE_POOLING_WINDOW * tmp_fs)
#     n = arr.shape[0]
#
#     trimmed_len = n - (n % window_size)
#     if trimmed_len == 0:
#         return np.array([], dtype=bool)
#     trimmed_arr = arr[:trimmed_len]
#
#     mean_vals = trimmed_arr.reshape(-1, window_size).astype(np.float32).mean(axis=1)
#     pooled_result = mean_vals >= threshold
#     return pooled_result


def row_col_from_n_subplots(n_subplots: int) -> Tuple[int, int]:
    n_row = int(np.floor(np.sqrt(n_subplots)))
    if n_subplots == n_row * n_row:
        return n_row, n_row
    elif n_subplots <= n_row*(n_row+1):
        return n_row, n_row+1
    else:
        return n_row+1, n_row+1


def linear_reg(ax: matplotlib.axes.Axes, x_data, y_data, n_pts: int = 100, **kwargs):
    slope, intercept = np.polyfit(x_data, y_data, 1)
    regression_x = np.linspace(min(x_data), max(x_data), n_pts)
    regression_y = slope * regression_x + intercept
    ax.plot(regression_x, regression_y, **kwargs)


def set_size(w, h, fig):
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    fig.set_size_inches(figw, figh)


def bin_average(times: np.ndarray | List, values: np.ndarray | List, bin_width: float, overlap=0.5):
    times = np.array(times)
    values = np.array(values)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]

    step = bin_width * (1 - overlap)
    centers = np.arange(times.min() + bin_width / 2, times.max() - bin_width / 2, step)

    bin_times, bin_avgs, bin_vars = [], [], []
    for c in centers:
        mask = (times >= c - bin_width / 2) & (times <= c + bin_width / 2)
        if np.any(mask):
            bin_times.append(c)
            bin_avgs.append(nan_mean(values[mask]))
            bin_vars.append(nan_sem(values[mask]))
    return np.array(bin_times), np.array(bin_avgs), np.array(bin_vars)


def bin_count(times: np.ndarray | List, bin_width: float, overlap=0.5):
    times = np.array(sorted(times))

    step = bin_width * (1 - overlap)
    centers = np.arange(times.min() + bin_width / 2, times.max() - bin_width / 2, step)

    bin_times, bin_avgs = [], []
    for c in centers:
        mask = (times >= c - bin_width / 2) & (times <= c + bin_width / 2)
        bin_times.append(c)
        bin_avgs.append(np.sum(mask))
    return np.array(bin_times), np.array(bin_avgs)

