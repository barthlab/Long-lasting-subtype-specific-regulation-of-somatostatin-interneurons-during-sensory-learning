from src import *


if __name__ == "__main__":
    for exp_id in ("Calb2_SAT", "Calb2_PSE"):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature_names = feature_prepare(mitten_feature)
        basic_evoked_response_and_fold_change(mitten_feature)
        if exp_id == "Calb2_SAT":
            visualize.plot_plasticity_manifold(
                mitten_feature, days1="ACC456", days2="SAT123",
                save_name=path.join("3_plasticity_manifold", f"{exp_id}_plasticity_calb2_intensity1"),
            )
            visualize.plot_plasticity_manifold(
                mitten_feature, days1="ACC456", days2="SAT8910",
                save_name=path.join("3_plasticity_manifold", f"{exp_id}_plasticity_calb2_intensity2.png"),
            )
        else:
            visualize.plot_plasticity_manifold(
                mitten_feature, days1="ACC456", days2="PSE12",
                save_name=path.join("3_plasticity_manifold", f"{exp_id}_plasticity_calb2_intensity1.png"),
            )
            visualize.plot_plasticity_manifold(
                mitten_feature, days1="ACC456", days2="PSE45",
                save_name=path.join("3_plasticity_manifold", f"{exp_id}_plasticity_calb2_intensity2.png"),
            )
