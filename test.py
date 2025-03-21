from src import *


if __name__ == "__main__":
    mitten = Mice(exp_id="Calb2_SAT", mice_id="M081")
    for cs in mitten.cell_sessions:
        if cs.cell_id == 9 and cs.day_id is SatDay.ACC5 and cs.session_id == 4 and cs.fov_id == 1:
            visualize.plot_cell_session(cs, "test.jpg")