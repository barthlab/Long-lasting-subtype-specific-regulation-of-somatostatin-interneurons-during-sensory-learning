from enum import Enum
from dataclasses import dataclass
from typing import Union, Dict, Tuple


# event label
class EventType(Enum):
    Puff = 1
    Blank = 0


STR2EVENT = {
    "real": EventType.Puff,
    "fake": EventType.Blank,
    "Puff": EventType.Puff,
    "Blank": EventType.Blank,
}


# spontaneous block
class BlockType(Enum):
    InterBlock = 0
    PreBlock = 1
    PostBlock = 2


# day enum
class SatDay(Enum):
    ACC1 = 0
    ACC2 = 1
    ACC3 = 2
    ACC4 = 3
    ACC5 = 4
    ACC6 = 5
    SAT1 = 6
    SAT2 = 7
    SAT3 = 8
    SAT4 = 9
    SAT5 = 10
    SAT6 = 11
    SAT7 = 12
    SAT8 = 13
    SAT9 = 14
    SAT10 = 15


SAT_ACC_DAYS = (SatDay(i) for i in range(6))
SAT_SAT_DAYS = (SatDay(i) for i in range(6, 16))
SAT_ALL_DAYS = (SatDay(i) for i in range(16))
SAT_PLOT_DAYS = (SatDay.ACC2, SatDay.ACC3, SatDay.ACC4, SatDay.ACC5, SatDay.ACC6,
                 SatDay.SAT1, SatDay.SAT2, SatDay.SAT3, SatDay.SAT4, SatDay.SAT5,)


class PseDay(Enum):
    ACC1 = 0
    ACC2 = 1
    ACC3 = 2
    ACC4 = 3
    ACC5 = 4
    ACC6 = 5
    PSE1 = 6
    PSE2 = 7
    PSE3 = 8
    PSE4 = 9
    PSE5 = 10
    PSE6 = 11
    PSE7 = 12
    PSE8 = 13
    PSE9 = 14
    PSE10 = 15


PSE_ACC_DAYS = (PseDay(i) for i in range(6))
PSE_PSE_DAYS = (PseDay(i) for i in range(6, 16))
PSE_ALL_DAYS = (PseDay(i) for i in range(16))
PSE_PLOT_DAYS = (PseDay.ACC4, PseDay.ACC5, PseDay.ACC6, PseDay.PSE1, PseDay.PSE5, PseDay.PSE9)

DayType = Union[PseDay, SatDay]


ADV_SAT: Dict[str, Tuple[DayType, ...]] = {
    "ACC123": tuple(SatDay(i) for i in range(3)),
    "ACC456": tuple(SatDay(i) for i in range(3, 6)),
    "SAT123": tuple(SatDay(i) for i in range(6, 9)),
    "SAT456": tuple(SatDay(i) for i in range(9, 12)),
    "SAT789": tuple(SatDay(i) for i in range(12, 15)),
    "SAT12": tuple(SatDay(i) for i in range(6, 8)),
    "SAT45": tuple(SatDay(i) for i in range(9, 11)),
    "SAT56": tuple(SatDay(i) for i in range(10, 12)),
    "SAT910": tuple(SatDay(i) for i in range(14, 16)),
}
ADV_SAT.update({SatDay(i).name: tuple((SatDay(i),)) for i in range(16)})

ADV_PSE: Dict[str, Tuple[DayType, ...]] = {
    "ACC123": tuple(PseDay(i) for i in range(3)),
    "ACC456": tuple(PseDay(i) for i in range(3, 6)),
    "PSE123": tuple(PseDay(i) for i in range(6, 9)),
    "PSE456": tuple(PseDay(i) for i in range(9, 12)),
    "PSE789": tuple(PseDay(i) for i in range(12, 15)),
    "PSE12": tuple(PseDay(i) for i in range(6, 8)),
    "PSE45": tuple(PseDay(i) for i in range(9, 11)),
    "PSE56": tuple(PseDay(i) for i in range(10, 12)),
    "PSE910": tuple(PseDay(i) for i in range(14, 16)),
}
ADV_PSE.update({PseDay(i).name: tuple((PseDay(i),)) for i in range(16)})


# dataclass identifier
@dataclass(frozen=True, order=True)
class CellUID:
    exp_id: str
    mice_id: str
    fov_id: int
    cell_id: int

    def in_short(self) -> str:
        return f"{self.exp_id} {self.mice_id} FOV{self.fov_id} Cell{self.cell_id}"


@dataclass(frozen=True, order=True)
class MiceUID:
    exp_id: str
    mice_id: str

    def in_short(self) -> str:
        return f"{self.mice_id}"


@dataclass(frozen=True, order=True)
class SessionUID:
    exp_id: str
    mice_id: str
    fov_id: int
    day_id: SatDay | PseDay
    session_in_day: int


class CellType(Enum):
    Unknown = -1
    Calb2_Neg = 0
    Calb2_Pos = 1
    Put_Calb2_Neg = 2
    Put_Calb2_Pos = 3



