from dataclasses import dataclass, field, MISSING
from typing import List, Callable, Optional, Dict
from functools import cached_property
import numpy as np
from scipy.ndimage import percentile_filter
import os
import os.path as path

from src.basic.utils import *
from src.config import *


###################
# data container ##
###################
@dataclass
class TimeSeries:
    v: np.ndarray
    t: np.ndarray
    drop: np.ndarray
    origin_t: float | None

    def __post_init__(self):
        assert self.v.ndim == 1 == self.t.ndim == self.drop.ndim, "values, drop, times should be 1-d array"
        assert len(self.v) == len(self.t) == len(self.drop), "values, drop, times have different length"
        if self.origin_t is None and len(self.t) > 0:
            self.origin_t = float(self.t[0])

    @property
    def t_aligned(self):
        return self.t - self.origin_t

    @cached_property
    def num_points(self):
        return len(self.v)

    def segment(self, start_t: float = 0, end_t: float = SESSION_DURATION,
                new_origin_t: float | None = None, relative_flag: bool = False) -> "TimeSeries":
        assert end_t > start_t
        segment_start_index = np.searchsorted(self.t, start_t if not relative_flag else start_t + self.origin_t)
        segment_end_index = np.searchsorted(self.t, end_t if not relative_flag else end_t + self.origin_t)
        if new_origin_t is not None and relative_flag:
            new_origin_t += self.origin_t
        return TimeSeries(
            v=self.v[segment_start_index: segment_end_index],
            t=self.t[segment_start_index: segment_end_index],
            origin_t=None if new_origin_t is None else new_origin_t,
            drop=self.drop[segment_start_index: segment_end_index]
        )


@dataclass
class Events:
    t: list | np.ndarray
    label: List[EventType]
    origin_t: float | None

    def __post_init__(self):
        self.t = np.array(self.t)
        assert self.t.ndim == 1, "time must be 1-d array"
        assert len(self.t) == len(self.label), f"time {len(self.t)} and label {len(self.label)} have different length"
        if self.origin_t is None and len(self.t) > 0:
            self.origin_t = self.t[0]

    @property
    def t_aligned(self):
        return self.t - self.origin_t

    @cached_property
    def num_events(self):
        return len(self.t)

    def segment(self, start_t: float = 0, end_t: float = SESSION_DURATION,
                new_origin_t: float | None = None, relative_flag: bool = False) -> "Events":
        assert end_t > start_t
        segment_start_index = np.searchsorted(self.t, start_t if not relative_flag else start_t + self.origin_t)
        segment_end_index = np.searchsorted(self.t, end_t if not relative_flag else end_t + self.origin_t)
        if new_origin_t is not None and relative_flag:
            new_origin_t += self.origin_t
        return Events(
            t=self.t[segment_start_index: segment_end_index],
            origin_t=None if new_origin_t is None else new_origin_t,
            label=self.label[segment_start_index: segment_end_index]
        )


def load_2p(mat_dict: dict) -> (np.ndarray, np.ndarray, np.ndarray):
    # Extract necessary data
    ops = mat_dict['ops'][0][0]
    F = mat_dict['F'].copy()
    Fneu = mat_dict['Fneu'].copy()
    num_neurites, total_frames = F.shape

    def corrupted_indices(axis_offset: np.ndarray) -> np.ndarray:
        reshape_offset = np.array(axis_offset).reshape(-1, SESSION_FRAMES_2P)
        normalized_offset = reshape_offset - np.mean(reshape_offset, axis=-1, keepdims=True)
        return np.array(np.abs(normalized_offset.reshape(-1)) > FRAME_LOST_THRESHOLD)

    # drop frames
    corrupted_frames = np.argwhere(corrupted_indices(ops['xoff']) | corrupted_indices(ops['yoff']))[:, 0]
    block_start, dropped_frames = 0, np.zeros(total_frames)
    for i, cur_frame in enumerate(corrupted_frames):
        if (i == len(corrupted_frames) - 1) or (corrupted_frames[i + 1] > cur_frame + 1):
            start_frame = corrupted_frames[block_start]
            block_len = i - block_start + 1

            if (block_len <= FRAME_INTERPOLATE_THRESHOLD) and (cur_frame < total_frames-1):
                interp = np.linspace(0, 1, block_len + 2)[np.newaxis, 1:-1]
                F[:, start_frame:cur_frame + 1] = (interp * F[:, cur_frame + 1, np.newaxis] +
                                                   (1 - interp) * F[:, start_frame - 1, np.newaxis])
                Fneu[:, start_frame:cur_frame + 1] = (interp * Fneu[:, cur_frame + 1, np.newaxis] +
                                                      (1 - interp) * Fneu[:, start_frame - 1, np.newaxis])
            else:
                dropped_frames[start_frame:cur_frame + 1] = 1
            block_start = i + 1

    is_cell = np.argwhere(mat_dict['iscell'][:, 0] == 1)[:, 0]
    if DEBUG_FLAG:
        print(f"Found cell indices: {is_cell}")
    soma_fluorescence = F[is_cell] - 0.7 * Fneu[is_cell]
    return soma_fluorescence, dropped_frames, is_cell


@dataclass
class Trial:
    cell_id: int
    day_id: SatDay | PseDay
    session_id: int
    exp_id: str
    mice_id: str
    fov_id: int

    trial_id: int
    trial_type: EventType

    origin_t: float
    fluorescence: TimeSeries = field(repr=False)
    baseline_v: float = field(init=False)
    df_f0: TimeSeries = field(init=False, repr=False)
    drop_flag: bool = field(init=False)
    stims: Events = field(repr=False)

    def __post_init__(self):
        assert self.exp_id in EXP_LIST
        assert self.origin_t == self.fluorescence.origin_t == self.stims.origin_t

        # calculate df/f0
        self.baseline_v = np.mean(self.fluorescence.segment(*TRIAL_BASELINE_RANGE, relative_flag=True).v)
        self.df_f0 = TimeSeries(
            v=(self.fluorescence.v - self.baseline_v) / self.baseline_v,
            t=self.fluorescence.t,
            drop=self.fluorescence.drop,
            origin_t=self.fluorescence.origin_t
        )

        # calculate drop
        self.drop_flag = np.sum(self.fluorescence.drop) > 0

    @cached_property
    def cell_uid(self):
        return CellUID(cell_id=self.cell_id, exp_id=self.exp_id,
                       mice_id=self.mice_id, fov_id=self.fov_id)


@dataclass
class SpontBlock:
    cell_id: int
    day_id: SatDay | PseDay
    session_id: int
    exp_id: str
    mice_id: str
    fov_id: int

    block_type: BlockType
    block_id: int | None
    block_start: float = field(init=False)
    block_len: float = field(init=False)

    fluorescence: TimeSeries = field(repr=False)
    baseline_v: float = field(init=False)
    df_f0: TimeSeries = field(init=False, repr=False)
    stims: Events = field(repr=False)

    def __post_init__(self):
        assert self.exp_id in EXP_LIST
        assert (self.block_type in (BlockType.PreBlock, BlockType.PostBlock)) == (self.block_id is None)
        assert self.stims.num_events == 0

        # calculate df/f0
        self.baseline_v = np.percentile(self.fluorescence.v, q=GLOBAL_BASELINE_PERCENTILE)
        self.df_f0 = TimeSeries(
            v=(self.fluorescence.v - self.baseline_v) / self.baseline_v,
            t=self.fluorescence.t,
            drop=self.fluorescence.drop,
            origin_t=self.fluorescence.origin_t
        )

        # calculate block length
        self.block_start = float(self.fluorescence.t[0])
        self.block_len = float(self.fluorescence.t[-1] - self.fluorescence.t[0])  # s

    @cached_property
    def cell_uid(self):
        return CellUID(cell_id=self.cell_id, exp_id=self.exp_id,
                       mice_id=self.mice_id, fov_id=self.fov_id)


@dataclass
class CellSession:
    cell_id: int
    day_id: SatDay | PseDay
    session_id: int
    exp_id: str
    mice_id: str
    fov_id: int

    fluorescence: TimeSeries = field(repr=False)
    baseline: TimeSeries = field(init=False, repr=False)
    df_f0: TimeSeries = field(init=False, repr=False)
    stims: Events = field(repr=False)

    trials: List[Trial] = field(init=False, repr=False)
    spont_blocks: List[SpontBlock] = field(init=False, repr=False)

    def __post_init__(self):
        assert self.exp_id in EXP_LIST

        # calculate df/f0
        self.baseline = TimeSeries(
            v=percentile_filter(self.fluorescence.v, size=GLOBAL_BASELINE_WINDOW,
                                percentile=GLOBAL_BASELINE_PERCENTILE, ),
            t=self.fluorescence.t,
            drop=self.fluorescence.drop,
            origin_t=self.fluorescence.origin_t
        )
        self.df_f0 = TimeSeries(
            v=(self.fluorescence.v - self.baseline.v) / self.baseline.v,
            t=self.fluorescence.t,
            drop=self.fluorescence.drop,
            origin_t=self.fluorescence.origin_t
        )

        # prepare for split
        kwargs = {
            "cell_id": self.cell_id,
            "day_id": self.day_id,
            "session_id": self.session_id,
            "exp_id": self.exp_id,
            "mice_id": self.mice_id,
            "fov_id": self.fov_id,
        }

        # split trials
        self.trials = []
        for trial_id, (stim_time, stim_label) in enumerate(zip(self.stims.t, self.stims.label)):
            new_trial = Trial(
                **kwargs,
                trial_id=trial_id,
                trial_type=stim_label,
                origin_t=stim_time,
                fluorescence=self.fluorescence.segment(
                    start_t=stim_time + TRIAL_RANGE[0],
                    end_t=stim_time + TRIAL_RANGE[1],
                    new_origin_t=stim_time,
                ),
                stims=self.stims.segment(
                    start_t=stim_time + TRIAL_RANGE[0],
                    end_t=stim_time + TRIAL_RANGE[1],
                    new_origin_t=stim_time,
                )
            )
            self.trials.append(new_trial)

        # split spont blocks
        self.spont_blocks = [
            SpontBlock(
                **kwargs,
                block_type=BlockType.PreBlock,
                block_id=None,
                fluorescence=self.fluorescence.segment(end_t=self.stims.t[0] + BLOCK_PRE_TRIAL),
                stims=self.stims.segment(end_t=self.stims.t[0] + BLOCK_PRE_TRIAL),
            ), ]
        for block_id, stim_time in enumerate(self.stims.t):  # type: int, float
            block_end_time = stim_time + LAST_BLOCK_LEN if block_id == self.stims.num_events - 1 else (
                    self.stims.t[block_id + 1] + BLOCK_PRE_TRIAL)
            self.spont_blocks.append(SpontBlock(
                **kwargs,
                block_type=BlockType.InterBlock,
                block_id=block_id,
                fluorescence=self.fluorescence.segment(start_t=stim_time + BLOCK_POST_TRIAL, end_t=block_end_time),
                stims=self.stims.segment(start_t=stim_time + BLOCK_POST_TRIAL, end_t=block_end_time),
            ))
        self.spont_blocks.append(SpontBlock(
            **kwargs,
            block_type=BlockType.PostBlock,
            block_id=None,
            fluorescence=self.fluorescence.segment(start_t=self.stims.t[-1] + LAST_BLOCK_LEN),
            stims=self.stims.segment(start_t=self.stims.t[-1] + LAST_BLOCK_LEN),
        ), )

    @cached_property
    def cell_uid(self):
        return CellUID(cell_id=self.cell_id, exp_id=self.exp_id,
                       mice_id=self.mice_id, fov_id=self.fov_id)


#####################
## abstract docker ##
#####################
@dataclass
class Image:
    exp_id: str

    dataset: List[CellSession] = field(repr=False)

    cells_uid: List[CellUID] = field(init=False)
    days: List[SatDay | PseDay] = field(init=False)

    def __post_init__(self):
        self.cells_uid, self.days = [], []
        for single_cellsession in self.dataset:
            self.cells_uid.append(single_cellsession.cell_uid)
            self.days.append(single_cellsession.day_id)
        self.cells_uid = list(set(self.cells_uid))
        self.days = sorted(list(set(self.days)), key=lambda x: (0 if isinstance(x, SatDay) else 1, x.value))
        self.cells_uid.sort()

    @property
    def trials(self) -> List[Trial]:
        return sum([single_cs.trials for single_cs in self.dataset], [])

    @property
    def spont_blocks(self) -> List[SpontBlock]:
        return sum([single_cs.spont_blocks for single_cs in self.dataset], [])

    def cell_split(self) -> Dict[CellUID, "Image"]:
        split_dict = {single_uid: [] for single_uid in self.cells_uid}
        for single_cs in self.dataset:
            split_dict[single_cs.cell_uid].append(single_cs)
        for single_uid in self.cells_uid:
            split_dict[single_uid] = Image(exp_id=self.exp_id, dataset=split_dict[single_uid])
        return split_dict

    def day_split(self) -> Dict[SatDay | PseDay, "Image"]:
        split_dict = {single_day: [] for single_day in self.days}
        for single_cs in self.dataset:
            split_dict[single_cs.day_id].append(single_cs)
        for single_day in self.days:
            split_dict[single_day] = Image(exp_id=self.exp_id, dataset=split_dict[single_day])
        return split_dict

    def select(self, **criteria) -> "Image":
        def matches(one_data: CellSession) -> bool:
            return all(getattr(one_data, key, None) == value if not callable(value) else
                       value(getattr(one_data, key, None))
                       for key, value in criteria.items())
        sub_dataset = [d for d in self.dataset if matches(d)]
        return Image(exp_id=self.exp_id, dataset=sub_dataset)



@dataclass
class FOV:
    exp_id: str
    mice_id: str
    fov_id: int
    num_cell: int = field(init=False)
    num_session_per_day: int = field(init=False)

    cell_sessions: List[CellSession] = field(init=False, repr=False)

    def __post_init__(self):
        assert self.exp_id in EXP_LIST

        # check file existence
        for file_name in ("Arduino.xlsx", f"Arduino P6_{self.fov_id}.xlsx", "Arduino time point.xlsx",
                          f"Arduino time point P6_{self.fov_id}.xlsx", "Fall.mat"):
            if not path.exists(path.join(self.data_path, file_name)):
                print(f"File {path.join(self.data_path, file_name)} not found!")
        if DEBUG_FLAG:
            print(f"Loading {self.exp_id} {self.mice_id} FOV{self.fov_id}")

        # data load
        total_fluorescence, dropped_frames, cell_indices = load_2p(loadmat(path.join(self.data_path, "Fall.mat")))
        self.num_cell, num_total_frames = total_fluorescence.shape
        puff_times = read_xlsx(path.join(self.data_path, "Arduino time point.xlsx"), header=0)
        puff_types = read_xlsx(path.join(self.data_path, "Arduino.xlsx"), header=None)
        assert len(puff_types.keys()) == len(puff_times.keys()), \
            f"Arduino.xlsx file doesn't match Arduino time point.xlsx in {self.mice_id} FOV{self.fov_id}"

        # data extraction prepare
        num_raw_session = int(num_total_frames / SESSION_FRAMES_2P)
        self.num_session_per_day = 1 if num_raw_session <= 16 else 2
        days = EXP2DAY[self.exp_id]
        days_in_data, days_to_extract = list(days), list(days)
        if self.mice_id in LOST_DATA:
            if "corrupted" in LOST_DATA[self.mice_id]:
                for day_id in LOST_DATA[self.mice_id]["corrupted"]:
                    days_to_extract.remove(day_id)
            if "lost" in LOST_DATA[self.mice_id]:
                for day_id in LOST_DATA[self.mice_id]["lost"]:
                    days_in_data.remove(day_id)
        valid_days = sorted(list(set(days_in_data).intersection(set(days_to_extract))), key=lambda x: x.value)
        assert total_fluorescence.shape[1] == SESSION_FRAMES_2P * len(days_in_data) * self.num_session_per_day, \
            (f"fluorescence shape doesn't match: "
             f"{total_fluorescence.shape} != {SESSION_FRAMES_2P} x {len(days_in_data)} x {self.num_session_per_day}")

        # split cell session
        """
        Each Mice 
        -> FOV 
        -> Cell x Session 
        -> Cell x (Session per day x Day)
        """
        self.cell_sessions = []
        for extract_day in valid_days:  # type: PseDay | SatDay
            day_order = days_in_data.index(extract_day)
            for session_order_in_day in range(self.num_session_per_day):
                if DEBUG_FLAG:
                    print(f"Parsing {self.exp_id} {self.mice_id} FOV{self.fov_id} "
                          f"Day{extract_day.value} #{session_order_in_day}")
                num_session_before = self.num_session_per_day * day_order + session_order_in_day
                session_start_frame = num_session_before * SESSION_FRAMES_2P
                stim_data_index = 2 * day_order + session_order_in_day

                # extract puff info
                if TIMEPOINTS_MS2S_FLAG:
                    process_puff_time = puff_times[stim_data_index][:, 0] / 1000. - HDT
                else:
                    process_puff_time = puff_times[stim_data_index][:, 0] - HDT
                process_puff_types = [STR2EVENT[tmp_label] for tmp_label in puff_types[stim_data_index][:, 0]]

                # filter corrupted session
                current_session_uid = SessionUID(exp_id=self.exp_id, mice_id=self.mice_id, fov_id=self.fov_id,
                                                 day_id=extract_day, session_in_day=session_order_in_day)
                if current_session_uid in LOST_SESSION:
                    intact_trial_num = LOST_SESSION[current_session_uid]
                    process_puff_time = process_puff_time[:intact_trial_num]
                    process_puff_types = process_puff_types[:intact_trial_num]

                # create stims
                new_stims = Events(
                    t=process_puff_time,
                    label=process_puff_types,
                    origin_t=0,
                )
                for cell_order, cell_index in enumerate(cell_indices):
                    # create fluorescence
                    new_fluorescence = TimeSeries(
                        v=total_fluorescence[cell_order, session_start_frame: session_start_frame + SESSION_FRAMES_2P],
                        t=np.linspace(0, SESSION_DURATION, SESSION_FRAMES_2P),
                        drop=dropped_frames[session_start_frame: session_start_frame + SESSION_FRAMES_2P],
                        origin_t=0,
                    )

                    # make final cell session
                    self.cell_sessions.append(
                        CellSession(
                            cell_id=cell_index,
                            day_id=extract_day,
                            session_id=num_session_before,
                            exp_id=self.exp_id,
                            mice_id=self.mice_id,
                            fov_id=self.fov_id,

                            fluorescence=new_fluorescence,
                            stims=new_stims
                        ))

    @property
    def image(self) -> Image:
        return Image(exp_id=self.exp_id, dataset=self.cell_sessions)

    @cached_property
    def data_path(self) -> str:
        return path.join(ROOT_PATH, CALCIUM_DATA_PATH, self.exp_id, self.mice_id, f"FOV{self.fov_id}")

    @cached_property
    def mice_order(self) -> int:
        assert self.mice_id[0] == "M"
        return int(self.mice_id[1:])


@dataclass
class Mice:
    exp_id: str
    mice_id: str

    fovs: List[FOV] = field(init=False, repr=False)
    cell_sessions: List[CellSession] = field(init=False, repr=False)

    def __post_init__(self):
        assert self.exp_id in EXP_LIST
        print(f"Loading {self.exp_id} {self.mice_id}")

        # Load fovs
        self.fovs, self.cell_sessions = [], []
        for fov_name in os.listdir(self.data_path):
            assert fov_name[:3] == "FOV"
            fov_id = int(fov_name[3:])
            new_fov = FOV(
                exp_id=self.exp_id,
                mice_id=self.mice_id,
                fov_id=fov_id,
            )
            self.fovs.append(new_fov)

            # append child dataclasses
            self.cell_sessions += new_fov.cell_sessions

    @cached_property
    def data_path(self) -> str:
        return path.join(ROOT_PATH, CALCIUM_DATA_PATH, self.exp_id, self.mice_id)

    @cached_property
    def mice_order(self) -> int:
        assert self.mice_id[0] == "M"
        return int(self.mice_id[1:])

    @property
    def image(self) -> Image:
        return Image(exp_id=self.exp_id, dataset=self.cell_sessions)


@dataclass
class Experiment:
    exp_id: str
    mice: List[Mice] = field(init=False, repr=False)
    cell_sessions: List[CellSession] = field(init=False, repr=False)

    def __post_init__(self):
        assert self.exp_id in EXP_LIST

        # Load mice
        self.mice, self.cell_sessions = [], []
        for mice_name in os.listdir(self.data_path):
            assert mice_name[0] == "M" and mice_name[1:].isdigit()
            new_mice = Mice(
                exp_id=self.exp_id,
                mice_id=mice_name,
            )
            self.mice.append(new_mice)

            # append child dataclasses
            self.cell_sessions += new_mice.cell_sessions

    @cached_property
    def data_path(self) -> str:
        return path.join(ROOT_PATH, CALCIUM_DATA_PATH, self.exp_id)

    @property
    def image(self) -> Image:
        return Image(exp_id=self.exp_id, dataset=self.cell_sessions)
