from src import *


if __name__ == "__main__":
    mitten = Mice(exp_id="Calb2_SAT", mice_id="M081")
    for cs in mitten.cell_sessions:
        visualize.plot_cell_session(cs)