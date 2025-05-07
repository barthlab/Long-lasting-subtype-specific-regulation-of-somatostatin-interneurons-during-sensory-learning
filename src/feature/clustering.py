import numpy as np
from typing import List, Callable, Optional, Dict, Union, Tuple

from src.config import *
from src.basic.utils import *
from src.data_manager import *
from src.feature.feature_utils import *
from src.feature.feature_manager import *


def get_feature_vector(feature_db: FeatureDataBase,
                       selected_feature_names: List[str],
                       selected_days: str):
    all_feature_by_cells = combine_dicts(*[
        feature_db.get(single_feature, day_postfix=selected_days).standardized().by_cells
        for single_feature in selected_feature_names
    ])
    feature_vector = {}
    for cell_uid, feature_list in all_feature_by_cells.items():  # type: CellUID, list
        assert isinstance(cell_uid, CellUID)
        assert len(feature_list) == len(selected_feature_names)
        assert np.sum(np.isnan(feature_list)) == 0
        feature_vector[cell_uid] = np.array(feature_list)
    return feature_vector


# def get_waveform_vector(feature_db: FeatureDataBase,
#                         selected_feature_names: List[str],
#                         selected_days: str):

