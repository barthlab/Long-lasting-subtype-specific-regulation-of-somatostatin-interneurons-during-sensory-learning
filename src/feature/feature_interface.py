import numpy as np
from typing import List, Callable, Optional, Dict, Union, Tuple

from src.config import *
from src.basic.utils import *
from src.data_manager import *
from src.feature.feature_utils import *
from src.feature.feature_manager import *


# CellDayWiseFeature Interface
def general_feature_interface(
        instance_list: Union[List[InstanceType]],
        metric_func: Callable[[InstanceType, ...], float],
        insert_name: str = None,
        instance_criteria: dict = None, func_criteria: dict = None) -> float:
    tmp_list = []
    instance_criteria = instance_criteria if instance_criteria is not None else {}
    func_criteria = func_criteria if func_criteria is not None else {}
    for single_instance in general_filter(instance_list, **instance_criteria):  # type: InstanceType
        trial_feature = metric_func(single_instance, **func_criteria)
        tmp_list.append(trial_feature)
        if insert_name is not None:
            if hasattr(single_instance, insert_name):
                raise AttributeError(f"Feature '{insert_name}' already exists on {single_instance}")
            setattr(single_instance, insert_name, trial_feature)
    assert len(tmp_list) > 0
    return nan_mean(tmp_list)


# # Feature funcs
def basic_evoked_response_and_fold_change(feature_db: FeatureDataBase, baseline_days: str = "ACC456"):
    feature_db.compute_DayWiseFeature(
        EVOKED_RESPONSE_FEATURE,
        func=lambda _single_cs: general_feature_interface(
            instance_list=_single_cs.trials, insert_name=EVOKED_RESPONSE_FEATURE,
            metric_func=element_feature_compute_trial_period_activity_peak,
            func_criteria=OPTIONS_TIME_RANGE["trial evoked period"],
            instance_criteria=OPTIONS_TRIAL_GROUP["stimulus trial only"]
        ))

    baseline = feature_db.get(EVOKED_RESPONSE_FEATURE, day_postfix=baseline_days)
    fold_change_feature_name = f"Fold-Change from {baseline_days}"
    feature_db.compute_DayWiseFeature(
        fold_change_feature_name,
        lambda single_cs: getattr(single_cs, EVOKED_RESPONSE_FEATURE) / baseline.get_cell(single_cs.cell_uid)
    )


def trial_wise_basic_features(feature_db: FeatureDataBase) -> List[str]:
    feature_name_list = []
    for time_range_name, time_range_option in OPTIONS_TIME_RANGE.items():
        for trial_group_name, trial_group_option in OPTIONS_TRIAL_GROUP.items():
            for element_func_name, element_func_option in OPTIONS_ELEMENT_FEATURE_TRIAL_PERIOD.items():
                tmp_feature_name = f"{element_func_name} || {time_range_name} || {trial_group_name}"
                feature_db.compute_DayWiseFeature(
                    feature_name=tmp_feature_name,
                    func=lambda _single_cs: general_feature_interface(
                        instance_list=_single_cs.trials,
                        metric_func=element_func_option,
                        insert_name=None,
                        instance_criteria=trial_group_option,
                        func_criteria=time_range_option
                    )
                )
                feature_name_list.append(tmp_feature_name)
    return feature_name_list


def trial_wise_response_probability_features(feature_db: FeatureDataBase) -> List[str]:
    feature_name_list = []
    for time_range_name, time_range_option in OPTIONS_TIME_RANGE.items():
        for trial_group_name, trial_group_option in OPTIONS_TRIAL_GROUP.items():
            for std_ratio_name, std_ratio_option in OPTIONS_STD_RATIO.items():
                tmp_feature_name = f"response prob || {time_range_name} || {trial_group_name} || {std_ratio_name}"
                feature_db.compute_DayWiseFeature(
                    feature_name=tmp_feature_name,
                    func=lambda _single_cs: general_feature_interface(
                        instance_list=_single_cs.trials,
                        metric_func=element_feature_compute_trial_period_responsive_probability,
                        insert_name=None,
                        instance_criteria=trial_group_option,
                        func_criteria={"trail_baseline_std": _single_cs.trial_baseline_std,
                                       **time_range_option, **std_ratio_option}
                    )
                )
                feature_name_list.append(tmp_feature_name)
    return feature_name_list


def spont_block_wise_features(feature_db: FeatureDataBase) -> List[str]:
    feature_name_list = []
    for block_group_name, block_group_option in OPTIONS_SPONT_BLOCK_GROUP.items():
        for element_func_name, element_func_option in OPTIONS_ELEMENT_FEATURE_SPONT_BLOCK_OVERALL_STD_RELATED.items():
            for std_ratio_name, std_ratio_option in OPTIONS_STD_RATIO.items():
                tmp_feature_name = f"{element_func_name} || {block_group_name} || {std_ratio_name}"
                feature_db.compute_DayWiseFeature(
                    feature_name=tmp_feature_name,
                    func=lambda _single_cs: general_feature_interface(
                        instance_list=_single_cs.spont_blocks,
                        metric_func=element_func_option,
                        insert_name=None,
                        instance_criteria=block_group_option,
                        func_criteria={"overall_baseline_std": _single_cs.overall_baseline_std, **std_ratio_option}
                    )
                )
                feature_name_list.append(tmp_feature_name)
    for block_group_name, block_group_option in OPTIONS_SPONT_BLOCK_GROUP.items():
        for element_func_name, element_func_option in OPTIONS_ELEMENT_FEATURE_SPONT_BLOCK_BASIC.items():
            tmp_feature_name = f"{element_func_name} || {block_group_name}"
            feature_db.compute_DayWiseFeature(
                feature_name=tmp_feature_name,
                func=lambda _single_cs: general_feature_interface(
                    instance_list=_single_cs.spont_blocks,
                    metric_func=element_func_option,
                    insert_name=None,
                    instance_criteria=block_group_option,
                    func_criteria={}
                )
            )
            feature_name_list.append(tmp_feature_name)
    return feature_name_list


def feature_prepare(feature_db: FeatureDataBase, overwrite=False):
    if (not overwrite) and (feature_db.archive_exists()):
        total_feature_names = feature_db.load_feature()
    else:
        print(f"Computing features...")
        basic_feature_names = trial_wise_basic_features(feature_db)
        response_prob_feature_names = trial_wise_response_probability_features(feature_db)
        spont_block_feature_names = spont_block_wise_features(feature_db)
        total_feature_names = basic_feature_names+response_prob_feature_names+spont_block_feature_names
        feature_db.save_features()
    print(f"\nPreparation complete!\n{len(total_feature_names)} features: {total_feature_names}")
    return total_feature_names


