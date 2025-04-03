import numpy as np

from src.config import *
from src.basic.utils import *
from src.data_manager import *

from dataclasses import dataclass, field, MISSING
from collections import defaultdict
from typing import List, Callable, Optional, Dict, Union, Tuple
from functools import cached_property


@dataclass
class CellWiseFeature:
    feature_name: str
    cells_uid: List[CellUID] = field(repr=False)
    value: List[float] | np.ndarray = field(repr=False)

    _data: Dict[CellUID, float] = field(init=False, repr=False)

    def __post_init__(self):
        assert len(self.cells_uid) == len(self.value)
        self._data = {}
        for cell_cnt, cell_uid in enumerate(self.cells_uid):
            self._data[cell_uid] = self.value[cell_cnt]

    def get_cell(self, cell_uid: CellUID) -> float:
        return self._data[cell_uid]


@dataclass
class CellDayWiseFeature:
    feature_name: str
    cells_uid: List[CellUID] = field(repr=False)
    days: List[DayType] = field(repr=False)
    value: np.ndarray = field(repr=False)

    _data: Dict[Tuple[CellUID, DayType], float] = field(init=False, repr=False)

    def __post_init__(self):
        assert self.value.shape == (len(self.cells_uid), len(self.days))
        self._data = {}
        for cell_cnt, cell_uid in enumerate(self.cells_uid):
            for day_cnt, day_id in enumerate(self.days):
                self._data[(cell_uid, day_id)] = self.value[cell_cnt][day_cnt]

    def get_cell(self, cell_uid: CellUID) -> Dict[DayType, float]:
        return {day_id: self._data[(cell_uid, day_id)] for day_id in self.days}

    def get_raw_cell(self, cell_uid: CellUID) -> np.ndarray:
        return np.array([self._data[(cell_uid, day_id)] for day_id in self.days])

    def get_day(self, day_id: DayType) -> Dict[CellUID, float]:
        return {cell_uid: self._data[(cell_uid, day_id)] for cell_uid in self.cells_uid}

    def get_raw_day(self, day_id: DayType) -> np.ndarray:
        return np.array([self._data[(cell_uid, day_id)] for cell_uid in self.cells_uid])


FeatureType = Union[CellWiseFeature, CellDayWiseFeature]


@dataclass
class FeatureDataBase:
    name: str
    ref_img: Image = field(repr=False)
    _features: List[FeatureType] = field(default=None, repr=False)

    def __post_init__(self):
        self._features = [] if self._features is None else self._features

        if self.ref_img.exp_id == "Calb2_SAT":
            self.load_Calb2()

    @property
    def feature_names(self) -> List[str]:
        return [single_feature.feature_name for single_feature in self._features]

    def load_Calb2(self):
        file_path = path.join(ROOT_PATH, FEATURE_DATA_PATH, CALB2_EXPRESSION_FILE_NAME)
        features = read_xlsx_sheet(file_path, header=0, sheet_id=0)
        num_cell, _ = features.shape

        # format check
        assert num_cell == len(self.cells_uid)
        for row_id in range(num_cell):
            assert features.iloc[row_id, 1] == f"cell {row_id+1}"

        self._features.append(CellWiseFeature("Calb2 Mean", self.cells_uid, features["Mean"]))
        self._features.append(CellWiseFeature("Calb2 SD", self.cells_uid, features["Mean"]))
        self._features.append(CellWiseFeature("Calb2 Max", self.cells_uid, features["Max"]))
        self._features.append(CellWiseFeature("Calb2", self.cells_uid, np.array(features["Mean"] >= CALB2_THRESHOLD)))

    def compute_DayWiseFeature(self, feature_name: str, func: Callable[[CellSession], float]):
        if feature_name in self.feature_names:
            return
        tmp_dict = {(cell_uid, day_id): [] for cell_uid in self.cells_uid for day_id in self.days}
        for single_cs in self.ref_img.dataset:
            tmp_dict[(single_cs.cell_uid, single_cs.day_id)].append(func(single_cs))
        final_matrix = np.full((len(self.cells_uid), len(self.days)), np.nan)
        for cell_cnt, cell_uid in enumerate(self.cells_uid):
            for day_cnt, day_id in enumerate(self.days):
                final_matrix[cell_cnt, day_cnt] = nan_mean(tmp_dict[(cell_uid, day_id)])
        self._features.append(CellDayWiseFeature(feature_name, self.cells_uid, self.days, final_matrix))

    def compute_CellWiseFeature(self, feature_name: str, func: Callable[[CellUID, "FeatureDataBase"], float]):
        if feature_name in self.feature_names:
            return
        final_array = np.full((len(self.cells_uid),), np.nan)
        for cell_cnt, cell_uid in enumerate(self.cells_uid):
            final_array[cell_cnt] = func(cell_uid, self)
        self._features.append(CellWiseFeature(feature_name, self.cells_uid, final_array))

    def get(self, feature_name: str) -> FeatureType:
        assert feature_name in self.feature_names
        for single_feature in self._features:
            if single_feature.feature_name == feature_name:
                return single_feature

    @cached_property
    def cell_types(self) -> Dict[CellUID, CellType]:
        if "Calb2" in self.feature_names:
            calb2_type = self.get("Calb2")
            return {cell_uid: CellType(calb2_type.get_cell(cell_uid)) for cell_uid in self.cells_uid}
        else:
            return {cell_uid: CellType.Unknown for cell_uid in self.cells_uid}

    @cached_property
    def cells_uid(self) -> List[CellUID]:
        return self.ref_img.cells_uid

    @cached_property
    def days(self) -> List[DayType]:
        return self.ref_img.days


def compute_evoked_peak(single_cs: CellSession) -> float:
    tmp_list = []
    for single_trial in single_cs.trials:
        if (not single_trial.drop_flag) and (single_trial.trial_type is EventType.Puff):
            trial_clip = single_trial.df_f0.segment(0, TEST_EVOKED_PERIOD, relative_flag=True).v
            tmp_list.append(nan_mean(trial_clip))
    return nan_mean(tmp_list)


def compute_response_prob(single_cs: CellSession) -> float:
    tmp_list = []
    for single_trial in single_cs.trials:
        if (not single_trial.drop_flag) and (single_trial.trial_type is EventType.Puff):
            evoked_period = single_trial.df_f0.segment(0, TEST_EVOKED_PERIOD, relative_flag=True).v
            baseline_period = single_trial.df_f0.segment(*TRIAL_BASELINE_RANGE, relative_flag=True).v
            tmp_list.append(np.mean(evoked_period) >= TEST_STD_RATIO * np.std(baseline_period))
    return nan_mean(tmp_list)


def compute_average_feature(cell_uid: CellUID, features: FeatureDataBase, daywise_feature_name: str,
                            selected_days_cnt: List[int]) -> float:
    chosen_feature = features.get(daywise_feature_name)
    assert isinstance(chosen_feature, CellDayWiseFeature)
    cell_feature_dict = chosen_feature.get_cell(cell_uid)
    return nan_mean([cell_feature_dict[day_id] for day_id in features.days if day_id.value in selected_days_cnt])

