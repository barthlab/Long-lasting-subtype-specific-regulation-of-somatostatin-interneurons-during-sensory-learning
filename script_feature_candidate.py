from src import *


def _single_feature_visualization(feature_db: FeatureDataBase, feature_names_to_plot: List[str]):
    assert not feature_db.Ai148_flag
    for single_feature_name in feature_names_to_plot:
        plot_feature.plot_single_feature_Calb2(
            feature_db, save_name=path.join("feature", feature_db.name, feature_db.ref_img.exp_id,
                                            feature_name_to_file_name(single_feature_name)),
            feature_name=single_feature_name,
        )


if __name__ == "__main__":
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        mitten_feature_names = feature_prepare(mitten_feature)
        _single_feature_visualization(mitten_feature, mitten_feature_names)
        mitten_pvalues = mitten_feature.pvalue_ttest_ind_calb2(mitten_feature_names)
        plot_feature.plot_feature_distribution_calb2(
            mitten_feature, save_name=path.join(
                "feature", mitten_feature.name, mitten_feature.ref_img.exp_id + "_features.png"),
            sorted_p_value_dict=mitten_pvalues
        )
        exit()

