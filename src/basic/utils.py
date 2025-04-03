import pandas as pd
from scipy.io import loadmat, savemat
from typing import Optional, List
import numpy as np
import scipy.signal as signal
from scipy import stats
from src.config import *


def read_xlsx(table_dir: str, header: None | int) -> dict[int, np.ndarray]:
    xl_file = pd.ExcelFile(table_dir)
    xlsx_dict = {sheet_id: xl_file.parse(sheet_name, header=header).to_numpy()
                 for sheet_id, sheet_name in enumerate(xl_file.sheet_names)}
    return xlsx_dict


def read_xlsx_sheet(table_dir: str, header: None | int, sheet_id: int = 0) -> pd.DataFrame:
    xl_file = pd.ExcelFile(table_dir)
    return xl_file.parse(sheet_name=xl_file.sheet_names[sheet_id], header=header)


def nan_mean(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmean(replaced_a, **kwargs)


def nan_median(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmedian(replaced_a, **kwargs)


def nan_std(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanstd(replaced_a, **kwargs)


def nan_max(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmax(replaced_a, **kwargs)


def nan_min(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmin(replaced_a, **kwargs)


def nan_sum(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nansum(replaced_a, **kwargs)


def low_pass(xs, fs, cutoff):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_ys = signal.filtfilt(b, a, xs)
    return filtered_ys


def high_pass(xs, fs, cutoff):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    filtered_ys = signal.filtfilt(b, a, xs)
    return filtered_ys


def general_filter(datalist: list, **criteria) -> list:
    def matches(one_data) -> bool:
        return all(getattr(one_data, key, None) == value if not callable(value) else
                   value(getattr(one_data, key, None))
                   for key, value in criteria.items())
    return [d for d in datalist if matches(d)]


def synchronize_time_series_data(
        times: List[np.ndarray], values: List[np.ndarray],
        elaborate: bool = False) -> (np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]):
    max_length = np.max([len(x) for x in times])
    min_time, max_time = np.min(np.concatenate(times)), np.max(np.concatenate(times))
    xs = np.linspace(min_time, max_time, int(max_length * 2))
    # for tmp_time, tmp_value in zip(times, values):
    #     print(tmp_value.shape, tmp_time.shape)
    interpolated_values = [np.interp(xs, tmp_time, tmp_value)
                           for tmp_time, tmp_value in zip(times, values)]
    mean_value = np.mean(np.array(interpolated_values), axis=0)
    sem_value = np.std(np.array(interpolated_values), axis=0) / np.sqrt(len(interpolated_values))
    return xs, mean_value, sem_value, interpolated_values

