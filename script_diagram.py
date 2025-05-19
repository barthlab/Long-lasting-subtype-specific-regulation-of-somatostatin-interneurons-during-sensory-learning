from src import *


_tmp_diagram_path = path.join("diagram", )


if __name__ == "__main__":
    for exp_id in ("Calb2_SAT",):
        mitten_fov = FOV(exp_id=exp_id, mice_id="M088", fov_id=1)
        mitten_cs = general_filter(mitten_fov.cell_sessions, cell_id=0, day_id=SatDay.ACC6,)[0]
        visualize.plot_cell_session(mitten_cs, save_name=path.join(_tmp_diagram_path, "overview.png"))
        plot_diagram.plot_diagram_large_view(mitten_cs, save_name=path.join(_tmp_diagram_path, "large_view.png"))
        plot_diagram.plot_diagram_trials(mitten_cs, save_name=path.join(_tmp_diagram_path, "trials.png"))
        plot_diagram.plot_diagram_spont_blocks(mitten_cs, save_name=path.join(_tmp_diagram_path, "spont_blocks.png"))
        plot_diagram.plot_diagram_trials_colored(mitten_cs, save_name=path.join(_tmp_diagram_path, "trials_colored.png"))
        plot_diagram.plot_diagram_spont_blocks_colored(mitten_cs, save_name=path.join(_tmp_diagram_path, "spont_blocks_colored.png"))
        plot_diagram.plot_diagram_spont_blocks_event_detection(mitten_cs, save_name=path.join(_tmp_diagram_path, "spont_blocks_event_detect.png"))
        plot_diagram.plot_diagram_trial_periods(mitten_cs, save_name=path.join(_tmp_diagram_path, "trial_periods.png"))
        plot_diagram.plot_diagram_prob_example(mitten_cs, save_name=path.join(_tmp_diagram_path, "trials_prob_example.png"))
        plot_diagram.plot_diagram_trial_representative(mitten_cs, save_name=path.join(_tmp_diagram_path, "trials_representative.png"))
        plot_diagram.plot_diagram_spont_representative(mitten_cs, save_name=path.join(_tmp_diagram_path, "spont_representative.png"))


