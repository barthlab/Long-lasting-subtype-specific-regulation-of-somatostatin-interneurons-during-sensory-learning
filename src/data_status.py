
from src.basic.terminology import *


# data lost
# reason: imaging software crashed
EXP_LOST_DAYS = {
    "Calb2_PSE": [PseDay.PSE6, PseDay.PSE7, PseDay.PSE8, PseDay.PSE9, PseDay.PSE10, ]
}
LOST_FOV = {
    FovUID(exp_id="Calb2_SAT", mice_id="M088", fov_id=1): [SatDay.SAT7, ],
    FovUID(exp_id="Calb2_SAT", mice_id="M088", fov_id=2): [SatDay.SAT7, ],
    FovUID(exp_id="Calb2_SAT", mice_id="M088", fov_id=3): [SatDay.SAT7, ],
    FovUID(exp_id="Ai148_SAT", mice_id="M017", fov_id=1): [SatDay.ACC1, ],

    # new Calb2_PSE dataset
    FovUID(exp_id="Calb2_PSE", mice_id="M201", fov_id=3): [PseDay.ACC1, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M201", fov_id=4): [PseDay.ACC1, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M202", fov_id=3): [PseDay.ACC1, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M204", fov_id=2): [PseDay.ACC1, PseDay.ACC2, PseDay.ACC3, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M204", fov_id=3): [PseDay.ACC1, PseDay.ACC2, PseDay.ACC3, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M205", fov_id=4): [PseDay.ACC1, PseDay.ACC2, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M206", fov_id=1): [PseDay.PSE1, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M206", fov_id=2): [PseDay.PSE1, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M206", fov_id=3): [PseDay.ACC1, PseDay.ACC2, PseDay.PSE1, ],
    FovUID(exp_id="Calb2_PSE", mice_id="M206", fov_id=4): [PseDay.ACC1, PseDay.ACC2, PseDay.PSE1, ],
}


# data to drop
# reason: data content corruption / timing file lost
UNWANTED_FOV = {
    FovUID(exp_id="Calb2_SAT", mice_id="M087", fov_id=2): [SatDay.SAT9, SatDay.SAT10],
    FovUID(exp_id="Calb2_SAT", mice_id="M087", fov_id=3): [SatDay.SAT9, SatDay.SAT10],
    FovUID(exp_id="Calb2_SAT", mice_id="M087", fov_id=4): [SatDay.SAT9, SatDay.SAT10],

    FovUID(exp_id="Calb2_PSE", mice_id="M203", fov_id=3): [PseDay.ACC4, ],
}


# trail lost
# reason: wifi connection issue, old experiments sometimes will lose the last 1 trial.
LOST_SESSION = {
    SessionUID(exp_id="Calb2_SAT", mice_id="M085", fov_id=4, day_id=SatDay.SAT1, session_in_day=0): 15,
    SessionUID(exp_id="Calb2_SAT", mice_id="M099", fov_id=4, day_id=SatDay.SAT3, session_in_day=0): 19,
    SessionUID(exp_id="Ai148_PSE", mice_id="M037", fov_id=2, day_id=PseDay.ACC3, session_in_day=0): 19,
    SessionUID(exp_id="Ai148_PSE", mice_id="M037", fov_id=2, day_id=PseDay.ACC3, session_in_day=1): 19,
    SessionUID(exp_id="Ai148_PSE", mice_id="M046", fov_id=2, day_id=PseDay.ACC2, session_in_day=0): 19,
    SessionUID(exp_id="Ai148_PSE", mice_id="M046", fov_id=2, day_id=PseDay.ACC2, session_in_day=1): 19,
    SessionUID(exp_id="Ai148_SAT", mice_id="M023", fov_id=1, day_id=SatDay.SAT9, session_in_day=0): 19,
}


