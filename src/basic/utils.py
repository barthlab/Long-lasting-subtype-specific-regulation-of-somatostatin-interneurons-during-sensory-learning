import pandas as pd
from scipy.io import loadmat, savemat
from typing import Optional, List, Iterable
from collections import defaultdict
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
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmean(replaced_a, **kwargs)


def nan_median(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmedian(replaced_a, **kwargs)


def nan_std(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanstd(replaced_a, **kwargs)


def nan_sem(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanstd(replaced_a, **kwargs)/np.sqrt(np.count_nonzero(~np.isnan(replaced_a)))


def nan_max(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmax(replaced_a, **kwargs)


def nan_min(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmin(replaced_a, **kwargs)


def nan_sum(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if len(a) == 0:
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
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


def general_filter(datalist: Iterable, **criteria) -> list:
    def matches(one_data) -> bool:
        def check_criterion(key, value):
            retrieved_val = getattr(one_data, key, None)
            if callable(value):
                return value(retrieved_val)
            elif isinstance(value, tuple):
                return retrieved_val in value
            else:
                return retrieved_val == value
        return all(check_criterion(k, v) for k, v in criteria.items())
    return [d for d in datalist if matches(d)]


def combine_dicts(*dicts) -> dict:
    result = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            assert isinstance(value, float)
            result[key].append(value)
    return result


def average_dict(input_dict: dict) -> dict:
    return {k: nan_mean(v) for k, v in input_dict.items()}


def invert_nested_dict(input_dict: Dict[str, dict]) -> Dict[str, dict]:
    result_dict = defaultdict(dict)
    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            result_dict[k2][k1] = v2
    return result_dict


