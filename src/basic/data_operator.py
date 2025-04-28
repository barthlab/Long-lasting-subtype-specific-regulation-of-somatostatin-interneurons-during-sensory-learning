from dataclasses import dataclass, field, MISSING
from typing import List, Callable, Optional, Dict, Any
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict

from src.basic.utils import *
from src.config import *
from src.data_manager import *


def previous_interp(x_new, xp, fp):
    f_previous = interp1d(xp, fp, kind='previous', bounds_error=False, fill_value="extrapolate")
    return f_previous(x_new)


def sync_timeseries(list_timeseries: List[TimeSeries], scale_factor: float = 2) \
        -> Tuple[TimeSeries, TimeSeries, Tuple[np.ndarray, np.ndarray]]:
    list_times = [single_ts.t_aligned for single_ts in list_timeseries]
    list_values = [single_ts.v for single_ts in list_timeseries]
    max_length = np.max([len(x) for x in list_times])
    min_time, max_time = np.max([x[0] for x in list_times]), np.min([x[-1] for x in list_times])
    xs = np.linspace(min_time, max_time, int(max_length * scale_factor))
    interpolated_values = np.stack([previous_interp(xs, tmp_time, tmp_value)
                                    for tmp_time, tmp_value in zip(list_times, list_values)], axis=0)
    mean_value = TimeSeries(v=np.mean(interpolated_values, axis=0), t=xs, origin_t=0)
    sem_value = TimeSeries(v=np.std(interpolated_values, axis=0) / np.sqrt(len(xs)), t=xs, origin_t=0)

    return mean_value, sem_value, (xs, interpolated_values)


def by_cell2by_mice(dict_by_cell: Dict[CellUID, Any]) -> Dict[MiceUID, Any]:
    value_type = type(list(dict_by_cell.values())[0])

    dict_by_mice = defaultdict(list)
    for cell_uid, single_value in dict_by_cell.items():
        mice_uid = MiceUID(exp_id=cell_uid.exp_id, mice_id=cell_uid.mice_id)
        assert isinstance(single_value, value_type)
        dict_by_mice[mice_uid].append(single_value)
    if value_type is TimeSeries:
        result_dict = {mice_uid: sync_timeseries(list_of_ts)[0] for mice_uid, list_of_ts in dict_by_mice.items()}
    elif value_type in (float, np.float64):
        result_dict = {mice_uid: nan_mean(list_of_value) for mice_uid, list_of_value in dict_by_mice.items()}
    elif value_type in (dict, defaultdict):
        result_dict = {mice_uid: combine_dicts(*list_of_dict) for mice_uid, list_of_dict in dict_by_mice.items()}
    else:
        raise TypeError(f"Type not supported: {value_type}")
    return result_dict


def brightness(ts: TimeSeries) -> float:
    return np.mean(ts.segment(start_t=0, end_t=1., relative_flag=True).v)
