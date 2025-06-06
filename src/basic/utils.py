import os

import pandas as pd
from scipy.io import loadmat, savemat
from typing import Optional, List, Iterable
from collections import defaultdict, Counter
import numpy as np
import scipy.signal as signal
from scipy import stats
import json

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
    if (len(a) == 0) or (np.count_nonzero(~np.isnan(a)) == 0):
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmean(replaced_a, **kwargs)


def nan_median(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if (len(a) == 0) or (np.count_nonzero(~np.isnan(a)) == 0):
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmedian(replaced_a, **kwargs)


def nan_std(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if (len(a) == 0) or (np.count_nonzero(~np.isnan(a)) == 0):
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanstd(replaced_a, **kwargs)


def nan_sem(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if (len(a) == 0) or (np.count_nonzero(~np.isnan(a)) == 0):
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanstd(replaced_a, **kwargs)/np.sqrt(np.count_nonzero(~np.isnan(replaced_a)))


def nan_max(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if (len(a) == 0) or (np.count_nonzero(~np.isnan(a)) == 0):
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmax(replaced_a, **kwargs)


def nan_min(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if (len(a) == 0) or (np.count_nonzero(~np.isnan(a)) == 0):
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nanmin(replaced_a, **kwargs)


def nan_sum(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if (len(a) == 0) or (np.count_nonzero(~np.isnan(a)) == 0):
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return np.nansum(replaced_a, **kwargs)


def nan_zscore(a: np.ndarray | list, **kwargs) -> np.ndarray | float:
    if (len(a) == 0) or (np.count_nonzero(~np.isnan(a)) == 0):
        return 0 if REPLACE_BAD_VALUE_FLAG else np.nan
    else:
        replaced_a = np.nan_to_num(a, nan=0, posinf=1e10, neginf=-1e10) if REPLACE_BAD_VALUE_FLAG else np.copy(a)
        return (replaced_a - nan_mean(replaced_a, **kwargs)) / nan_std(replaced_a, **kwargs)


def nan_free(adata: np.ndarray | list) -> np.ndarray:
    array = np.array(adata)
    return array[~np.isnan(array)]


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


def general_select(datadict: dict, **criteria) -> dict:
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
    return {_k: _v for _k, _v in datadict.items() if matches(_k)}


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


def reverse_dict(input_dict: dict) -> dict:
    result_dict = defaultdict(list)
    for _k, _v in input_dict.items():
        result_dict[_v].append(_k)
    return result_dict


def decompose_cell_uid_list(cells_uid: List[CellUID]) -> dict:
    return {
        "exp_id": [cell_uid.exp_id for cell_uid in cells_uid],
        "mice_id": [cell_uid.mice_id for cell_uid in cells_uid],
        "fov_id": [cell_uid.fov_id for cell_uid in cells_uid],
        "cell_id": [cell_uid.cell_id for cell_uid in cells_uid],
    }


def synthesize_cell_uid_list(cell_element_dict: dict) -> List[CellUID]:
    for _k in ("exp_id", "mice_id", "fov_id", "cell_id"):
        assert _k in cell_element_dict
    cells_uid = [
        CellUID(exp_id=tmp_exp_id, mice_id=tmp_mice_id, fov_id=tmp_fov_id, cell_id=tmp_cell_id)
        for tmp_exp_id, tmp_mice_id, tmp_fov_id, tmp_cell_id in zip(
            cell_element_dict["exp_id"],
            cell_element_dict["mice_id"],
            cell_element_dict["fov_id"],
            cell_element_dict["cell_id"],
        )
    ]
    return cells_uid


def calb2_pos_neg_count(cells_uid: List[CellUID], cell_type_dict: Dict[CellUID, CellType], label_flag: bool = True,) \
        -> Tuple[int, int, float, str]:
    n_pos = Counter([cell_type_dict[cell_uid] for cell_uid in cells_uid]).get(CellType.Calb2_Pos, 0)
    n_neg = Counter([cell_type_dict[cell_uid] for cell_uid in cells_uid]).get(CellType.Calb2_Neg, 0)
    if n_neg + n_pos > 0:
        pos_ratio = 100*n_pos/(n_pos+n_neg)
        summary_string = f" +{n_pos} -{n_neg} {int(pos_ratio)}%" if label_flag else f" {int(pos_ratio)}%"
        return n_pos, n_neg, pos_ratio, summary_string
    else:
        return 0, 0, 0, ""


def json_dump(file_name: str, data_list: List[str]):
    os.makedirs(path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4)
    print(f"Data successfully saved to {file_name}")


def json_load(file_name: str) -> List[str]:
    with open(file_name, 'r', encoding='utf-8') as f:
        loaded_list = json.load(f)
    print(f"Data successfully loaded from {file_name}:")
    print(len(loaded_list), loaded_list)
    return loaded_list


def numpy_percentile_filter(input_array: np.ndarray, s: int, q: float) -> np.ndarray:
    if not isinstance(input_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if input_array.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")
    if not isinstance(s, int) or s <= 0:
        raise ValueError("Window size 's' must be a positive integer.")
    if not (0 <= q <= 100):
        raise ValueError("Percentile 'q' must be between 0 and 100.")

    if input_array.size == 0:
        return np.array([])

    pad_left = s // 2
    pad_right = s - 1 - pad_left
    arr_padded = np.pad(input_array, (pad_left, pad_right), mode='edge')
    shape_windows = (input_array.size, s)
    strides_windows = (arr_padded.strides[0], arr_padded.strides[0])

    windows = np.lib.stride_tricks.as_strided(arr_padded,
                                              shape=shape_windows,
                                              strides=strides_windows)
    result = np.percentile(windows, q, axis=1)

    return result


def simplify_day_str(day_str: str) -> str:
    prefix = day_str[:3]
    if len(day_str) == 4:
        return day_str
    elif len(day_str) == 5:
        return prefix + day_str[3] + "/" + day_str[4]
    elif len(day_str) == 6:
        return prefix + day_str[3] + "-" + day_str[5]
    elif len(day_str) == 7:
        assert day_str[-2:] == "10"
        return prefix + day_str[3] + "-10"
    else:
        raise ValueError(f"not parsable string: {day_str}")

