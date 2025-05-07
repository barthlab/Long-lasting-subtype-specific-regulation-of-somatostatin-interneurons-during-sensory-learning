from src import *


if __name__ == "__main__":
    mitten_behavior = BehaviorExperiment("Ai148_SAT")
    for mitten_mice in mitten_behavior.mice:
        plot_behavior.plot_heatmap_licks(
            mitten_mice,
            save_name=path.join("behavior", f"{mitten_mice.mice_id}_lick.png"))
