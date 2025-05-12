import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, MISSING
from collections import defaultdict
from typing import List, Callable, Optional, Dict, Union, Tuple
from functools import cached_property
import openpyxl

from src.config import *
from src.basic.utils import *
from src.data_manager import *
from src.feature.feature_utils import *


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

    @property
    def by_cells(self) -> Dict[CellUID, float]:
        return self._data

    def standardized(self) -> "CellWiseFeature":
        new_feature_name = self.feature_name + f"_standardized"
        mean_value = nan_mean(self.value)
        std_value = nan_std(self.value)
        assert std_value != 0
        return CellWiseFeature(feature_name=new_feature_name, cells_uid=self.cells_uid,
                               value=(np.array(self.value) - mean_value) / std_value)


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


def daywise_average(cell_uid: CellUID, chosen_feature: CellDayWiseFeature, selected_days: Tuple[DayType]) -> float:
    cell_feature_dict = chosen_feature.get_cell(cell_uid)
    return nan_mean([cell_feature_dict[day_id] for day_id in selected_days])


@dataclass
class FeatureDataBase:
    name: str
    ref_img: Image = field(repr=False)
    _features: List[FeatureType] = field(default=None, repr=False)
    prime: bool = field(default=True)

    def __post_init__(self):
        self._features = [] if self._features is None else self._features

        if self.ref_img.exp_id == "Calb2_SAT" and self.prime:
            self.load_Calb2()

    @property
    def feature_names(self) -> List[str]:
        return [single_feature.feature_name for single_feature in self._features]

    @cached_property
    def SAT_flag(self) -> bool:
        return EXP2DAY[self.ref_img.exp_id] is SatDay

    @cached_property
    def days_ref(self) -> Dict[str, Tuple[DayType, ...]]:
        return ADV_SAT if self.SAT_flag else ADV_PSE

    @cached_property
    def Ai148_flag(self) -> bool:
        return Ai148_FLAG[self.ref_img.exp_id]

    def load_Calb2(self):
        file_path = path.join(ROOT_PATH, FEATURE_DATA_PATH, CALB2_EXPRESSION_FILE_NAME)
        features = read_xlsx_sheet(file_path, header=0, sheet_id=0)
        num_cell, _ = features.shape

        # format check
        assert num_cell == len(self.cells_uid)
        for row_id in range(num_cell):
            assert features.iloc[row_id, 1] == f"cell {row_id+1}"

        self._features.append(CellWiseFeature("Calb2 Mean", self.cells_uid, features["Mean"]))
        # self._features.append(CellWiseFeature("Calb2 SD", self.cells_uid, features["Mean"]))
        # self._features.append(CellWiseFeature("Calb2 Max", self.cells_uid, features["Max"]))
        molecular_identity = np.array(features["Mean"] >= CALB2_THRESHOLD)
        self.compute_CellWiseFeature("Calb2",
                                     lambda cell_uid, _: float(molecular_identity[self.cells_idx[cell_uid]]))

    def compute_DayWiseFeature(self, feature_name: str, func: Callable[[CellSession], float]):
        if feature_name in self.feature_names:
            raise RuntimeError(f"Feature {feature_name} is already registered!")
        raw_feature_dict: Dict[Tuple[CellUID, DayType], List[float]] = defaultdict(list)
        for single_cs in self.ref_img.dataset:
            raw_feature_dict[(single_cs.cell_uid, single_cs.day_id)].append(func(single_cs))

        # average multiple sessions' result for each day each cell
        avg_matrix = np.full((len(self.cells_uid), len(self.days)), np.nan)
        for (cell_uid, day_id), values in raw_feature_dict.items():
            cell_idx, day_idx = self.cells_idx[cell_uid], self.days_idx[day_id]
            avg_matrix[cell_idx, day_idx] = nan_mean(raw_feature_dict[(cell_uid, day_id)])

        # insert feature into CellSession
        if self.prime:
            for single_cs in self.ref_img.dataset:
                if hasattr(single_cs, feature_name):
                    raise AttributeError(f"Feature '{feature_name}' already exists on {single_cs}")
                else:
                    setattr(single_cs, feature_name, avg_matrix[
                        self.cells_idx[single_cs.cell_uid], self.days_idx[single_cs.day_id]])
        self._features.append(CellDayWiseFeature(feature_name, self.cells_uid, self.days, avg_matrix))

    def compute_CellWiseFeature(self, feature_name: str, func: Callable[[CellUID, "FeatureDataBase"], float]):
        if feature_name in self.feature_names:
            raise RuntimeError(f"Feature {feature_name} has already been created!")
        final_array = np.full((len(self.cells_uid),), np.nan)
        for cell_cnt, cell_uid in enumerate(self.cells_uid):
            final_array[cell_cnt] = func(cell_uid, self)

        # insert feature into CellSession
        if self.prime:
            for single_cs in self.ref_img.dataset:
                if hasattr(single_cs, feature_name):
                    raise AttributeError(f"Feature '{feature_name}' already exists on {single_cs}")
                else:
                    setattr(single_cs, feature_name, final_array[self.cells_idx[single_cs.cell_uid]])
        self._features.append(CellWiseFeature(feature_name, self.cells_uid, final_array))

    def average_DayWise2CellWise(self, average_feature_names: List[str] = None,
                                 periods_dict: Dict[str, Tuple[DayType]] = None):
        if average_feature_names is None:
            average_feature_names = [single_feature.feature_name for single_feature in self._features
                                     if isinstance(single_feature, CellDayWiseFeature)]
        if periods_dict is None:
            periods_dict = self.days_ref
        for period_name, period_list in periods_dict.items():
            for target_feature_name in average_feature_names:
                target_feature = self.get(target_feature_name)
                assert isinstance(target_feature, CellDayWiseFeature)
                self.compute_CellWiseFeature(
                    f"{target_feature_name}_{period_name}",
                    lambda cell_uid, _: daywise_average(cell_uid, target_feature, period_list))

    def get(self, feature_name: str, day_postfix: str = None) -> FeatureType:
        assert feature_name in self.feature_names
        target_feature_name = feature_name if day_postfix is None else feature_name + f"_{day_postfix}"
        if target_feature_name not in self.feature_names:
            self.average_DayWise2CellWise(
                average_feature_names=[feature_name, ],
                periods_dict={day_postfix: self.days_ref[day_postfix]})
        for single_feature in self._features:
            if single_feature.feature_name == target_feature_name:
                return single_feature
        raise ValueError("Unknown Fetch Error!")

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
    def cells_idx(self) -> Dict[CellUID, int]:
        return {uid: i for i, uid in enumerate(self.cells_uid)}

    @cached_property
    def days(self) -> List[DayType]:
        return self.ref_img.days

    @cached_property
    def days_idx(self) -> Dict[DayType, int]:
        return {day: i for i, day in enumerate(self.days)}

    def select(self, new_name: str, **criteria) -> "FeatureDataBase":
        new_image = self.ref_img.select(**criteria)
        new_feature_db = FeatureDataBase(new_name, ref_img=new_image, prime=False)
        for old_feature in self._features:
            old_feature_name = old_feature.feature_name
            if isinstance(old_feature, CellDayWiseFeature):
                new_feature_db.compute_DayWiseFeature(
                    feature_name=old_feature_name, func=lambda single_cs: getattr(single_cs, old_feature_name))
            elif isinstance(old_feature, CellWiseFeature):
                new_feature_db.compute_CellWiseFeature(
                    feature_name=old_feature_name, func=lambda cell_uid, _: old_feature.get_cell(cell_uid))
        return new_feature_db

    @cached_property
    def feature_file_path(self) -> str:
        return path.join(FEATURE_EXTRACTED_PATH, f"{self.name}.xlsx")

    def save_features(self):
        print(f"Saving features to {self.feature_file_path}...")
        os.makedirs(path.dirname(self.feature_file_path), exist_ok=True)
        with pd.ExcelWriter(self.feature_file_path, engine='xlsxwriter') as writer:
            for single_feature in self._features:
                sheet_name = feature_name_to_file_name(single_feature.feature_name)
                print(f"Saving feature: {single_feature.feature_name}")
                prefix_dict = {
                    "feature_name": [single_feature.feature_name, ] + ["", ]*(len(single_feature.cells_uid)-1),
                    **decompose_cell_uid_list(single_feature.cells_uid),
                }
                if isinstance(single_feature, CellWiseFeature):
                    df = pd.DataFrame({**prefix_dict, 'value': single_feature.value})
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(single_feature, CellDayWiseFeature):
                    tmp_dict = {}
                    for day_id in single_feature.days:
                        tmp_dict[day_id.name] = single_feature.get_raw_day(day_id)
                    df = pd.DataFrame({**prefix_dict, **tmp_dict})
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    raise ValueError
        print(f"Feature sheets saved at: {self.feature_file_path}.")

    def load_feature(self) -> List[str]:
        print(f"Loading features from {self.feature_file_path}...")
        xls = pd.ExcelFile(self.feature_file_path, engine='openpyxl')
        loaded_features_names = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                raise ValueError
            feature_name = df["feature_name"][0]
            if feature_name in self.feature_names:
                continue
            print(f"Loading feature: {feature_name}")
            loaded_features_names.append(feature_name)
            tmp_cells_uid = synthesize_cell_uid_list({_k: df[_k].tolist()
                                                      for _k in ("exp_id", "mice_id", "fov_id", "cell_id")})
            assert tmp_cells_uid == self.cells_uid, \
                f"Different cells_uid\nStored File: {tmp_cells_uid}\nCurrent Feature Database: {self.cells_uid}"
            columns = df.columns.tolist()
            if "value" in columns:
                # CellWiseFeature
                value_array = df['value'].to_numpy(dtype=float)
                self._features.append(CellWiseFeature(
                    feature_name=feature_name, cells_uid=tmp_cells_uid, value=value_array))
            elif len(columns) > 6:
                # CellDayWiseFeature
                day_columns = columns[5:]
                tmp_days_id = [EXP2DAY[self.ref_img.exp_id][day_str] for day_str in day_columns]
                assert tmp_days_id == self.days, \
                    f"Different days\nStored File: {tmp_days_id}\nCurrent Feature Database: {self.days} "
                value_array = df[day_columns].to_numpy(dtype=float)
                self._features.append(CellDayWiseFeature(
                    feature_name=feature_name, cells_uid=tmp_cells_uid, days=tmp_days_id, value=value_array))
        print("Loading complete.")
        return loaded_features_names

    def archive_exists(self) -> bool:
        return path.exists(self.feature_file_path)

    def pvalue_ttest_ind_calb2(self, features_names: List[str], selected_days: str = "ACC456") -> Dict[str, float]:
        assert not self.Ai148_flag
        cell_types = reverse_dict(self.cell_types)
        p_value_dict = {}
        for single_feature_name in features_names:
            target_feature = self.get(feature_name=single_feature_name, day_postfix=selected_days).by_cells
            p_value_dict[single_feature_name] = stats.ttest_ind(
                [target_feature[cell_uid] for cell_uid in cell_types[CellType.Calb2_Pos]],
                [target_feature[cell_uid] for cell_uid in cell_types[CellType.Calb2_Neg]],
            ).pvalue

        return dict(sorted(p_value_dict.items(), key=lambda item: item[1]))

