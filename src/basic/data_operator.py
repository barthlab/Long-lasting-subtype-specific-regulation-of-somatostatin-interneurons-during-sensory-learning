from dataclasses import dataclass, field, MISSING
from typing import List, Callable, Optional, Dict, Any
import numpy as np

from src.basic.utils import *
from src.config import *
from src.data_manager import *


def sync_timeseries(list_timeseries: List[TimeSeries], scale_factor: float = 2) \
        -> Tuple[TimeSeries, TimeSeries, Tuple[np.ndarray, np.ndarray]]:
    list_times = [single_ts.t_aligned for single_ts in list_timeseries]
    list_values = [single_ts.v for single_ts in list_timeseries]
    max_length = np.max([len(x) for x in list_times])
    min_time, max_time = np.max([x[0] for x in list_times]), np.min([x[-1] for x in list_times])
    xs = np.linspace(min_time, max_time, int(max_length * scale_factor))
    interpolated_values = np.stack([np.interp(xs, tmp_time, tmp_value)
                                    for tmp_time, tmp_value in zip(list_times, list_values)], axis=0)
    mean_value = TimeSeries(v=np.mean(interpolated_values, axis=0), t=xs, origin_t=0)
    sem_value = TimeSeries(v=np.std(interpolated_values, axis=0) / np.sqrt(len(xs)), t=xs, origin_t=0)

    return mean_value, sem_value, (xs, interpolated_values)


