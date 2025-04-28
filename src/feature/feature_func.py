import numpy as np
from typing import List, Callable, Optional, Dict, Union, Tuple

from src.config import *
from src.basic.utils import *
from src.data_manager import *


# CellDayWiseFeature Interface
def general_feature_interface(
        instance_list: Union[List[InstanceType]],
        metric_func: Callable[[InstanceType, ...], float],
        insert_name: str = None, instance_criteria: dict = None, func_criteria: dict = None) -> float:
    tmp_list = []
    instance_criteria = instance_criteria if instance_criteria is not None else {}
    func_criteria = func_criteria if func_criteria is not None else {}
    for single_instance in general_filter(instance_list, **instance_criteria):  # type: InstanceType
        trial_feature = metric_func(single_instance, **func_criteria)
        tmp_list.append(trial_feature)
        if hasattr(single_instance, insert_name):
            raise AttributeError(f"Feature '{insert_name}' already exists on {single_instance}")
        else:
            setattr(single_instance, insert_name, trial_feature)
    assert len(tmp_list) > 0
    return nan_mean(tmp_list)


# element func
def feature_responsive_flag(single_trial: Trial,
                            start_t: float, end_t: float, ratio_std: float, noise_level: float) -> float:
    trial_clip = single_trial.df_f0.segment(start_t=start_t, end_t=end_t, relative_flag=True).v
    return np.max(trial_clip) >= ratio_std * noise_level


def feature_peak(single_trial: Trial,
                 start_t: float, end_t: float,) -> float:
    trial_clip = single_trial.df_f0.segment(start_t=start_t, end_t=end_t, relative_flag=True).v
    return np.max(trial_clip)


# Feature funcs
compute_trial_responsive: Callable[[CellSession], float] = lambda _single_cs: general_feature_interface(
    instance_list=_single_cs.trials, metric_func=feature_responsive_flag, insert_name="responsiveness",
    func_criteria={"start_t": 0, "end_t": TEST_EVOKED_PERIOD, "ratio_std": TEST_STD_RATIO,
                   "noise_level": _single_cs.noise_level},
)
compute_trial_evoked_peak: Callable[[CellSession], float] = lambda _single_cs: general_feature_interface(
    instance_list=_single_cs.trials, metric_func=feature_peak, insert_name="evoked_peak",
    func_criteria={"start_t": 0, "end_t": TEST_EVOKED_PERIOD,}, instance_criteria={"trial_type": EventType.Puff}
)


# def compute_peak(single_cs: CellSession, t_start: float, t_end: float) -> float:
#     tmp_list = []
#     for single_trial in single_cs.trials:
#         if (not single_trial.drop_flag) and (single_trial.trial_type is EventType.Puff):
#             trial_clip = single_trial.df_f0.segment(t_start, t_end, relative_flag=True)
#             tmp_list.append(nan_max(trial_clip.v))
#     return nan_mean(tmp_list)
#
#
# def compute_time2peak(single_cs: CellSession, t_start: float, t_end: float) -> float:
#     tmp_list = []
#     for single_trial in single_cs.trials:
#         if (not single_trial.drop_flag) and (single_trial.trial_type is EventType.Puff):
#             trial_clip = single_trial.df_f0.segment(t_start, t_end, relative_flag=True)
#             peak_frame = np.argmax(trial_clip.v)
#             tmp_list.append(trial_clip.t_aligned[peak_frame])
#     return nan_mean(tmp_list)
#
#
# def compute_auc(single_cs: CellSession, t_start: float, t_end: float) -> float:
#     tmp_list = []
#     for single_trial in single_cs.trials:  # type: Trial
#         if (not single_trial.drop_flag) and (single_trial.trial_type is EventType.Puff):
#             trial_clip = single_trial.df_f0.segment(t_start, t_end, relative_flag=True, new_origin_t=0)
#             trial_auc = np.trapz(trial_clip.v, x=trial_clip.t_aligned)
#             tmp_list.append(trial_auc)
#     return nan_mean(tmp_list)
#

# def compute_response_prob(single_cs: CellSession, t_start: float, t_end: float, ratio_std: float) -> float:
#     tmp_list = []
#     for single_trial in single_cs.trials:
#         if (not single_trial.drop_flag) and (single_trial.trial_type is EventType.Puff):
#             evoked_period = single_trial.df_f0.segment(t_start, t_end, relative_flag=True).v
#             baseline_period = single_trial.df_f0.segment(*TRIAL_BASELINE_RANGE, relative_flag=True).v
#             tmp_list.append(np.mean(evoked_period) >= ratio_std * np.std(baseline_period))
#     return nan_mean(tmp_list)