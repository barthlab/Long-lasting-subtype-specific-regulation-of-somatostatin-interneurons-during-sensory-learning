import pandas as pd
from scipy.io import loadmat, savemat
from typing import Optional
import numpy as np
from src.config import *
import scipy.signal as signal


def read_xlsx(table_dir: str, header: None | int) -> dict[int, np.ndarray]:
    xl_file = pd.ExcelFile(table_dir)
    xlsx_dict = {sheet_id: xl_file.parse(sheet_name, header=header).to_numpy()
                 for sheet_id, sheet_name in enumerate(xl_file.sheet_names)}
    return xlsx_dict


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

