import numpy as np
import os
import os.path as path

from src.data_manager import *
from src.basic.utils import *
from src.basic.data_operator import *
from src.config import *
from src.feature.feature_manager import *
from src.ploter.plotting_params import *


def oreo(ax: matplotlib.axes.Axes, trials: List[Trial], mean_kwargs: dict, fill_kwargs: dict,
         x_offset: float = 0, y_offset: float = 0):
    avg_y, sem_y, (avg_xs, _) = sync_timeseries([single_trial.df_f0 for single_trial in trials])
    ax.plot(avg_xs + x_offset, avg_y.v + y_offset, **mean_kwargs)
    ax.fill_between(avg_xs + x_offset, avg_y.v - sem_y.v + y_offset, avg_y.v + sem_y.v + y_offset, **fill_kwargs)


