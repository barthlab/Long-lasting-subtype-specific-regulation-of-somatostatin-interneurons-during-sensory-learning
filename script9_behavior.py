from src import *


if __name__ == "__main__":
    _exp = "Ai148_SAT"
    mitten_behavior = BehaviorExperiment(_exp)

    mitten = Experiment(exp_id=_exp)
    mitten_data = mitten.image
    mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
    mitten_feature_names = feature_prepare(mitten_feature)
    basic_evoked_response_and_fold_change(mitten_feature)

    for mitten_mice in mitten_behavior.mice:

        plot_behavior.plot_heatmap_licks(
            mitten_mice,
            save_name=path.join("behavior", f"{mitten_mice.mice_id}_lick.png"))
        p