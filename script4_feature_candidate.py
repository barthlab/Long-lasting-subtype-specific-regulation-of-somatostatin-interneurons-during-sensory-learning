from src import *


def _single_feature_visualization(feature_db: FeatureDataBase, p_value_dict: Dict[str, float]):
    assert not feature_db.Ai148_flag

    sorted_features_names = list(p_value_dict.keys())
    for feature_cnt, single_feature_name in enumerate(sorted_features_names):
        plot_feature.plot_single_feature_Calb2(
            feature_db, save_name=path.join(
                _tmp_save_name, feature_db.name, feature_db.ref_img.exp_id,
                f"feature{feature_cnt+1}_"+feature_name_to_file_name(single_feature_name)),
            feature_name=single_feature_name, sorted_features_names=sorted_features_names
        )


if __name__ == "__main__":
    _tmp_save_name = path.join("5_features")
    for exp_id in ("Calb2_SAT",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("justify_feature", mitten_data)

        mitten_feature_names = feature_prepare(mitten_feature)
        mitten_pvalues = mitten_feature.pvalue_ttest_ind_calb2(mitten_feature_names)
        json_dump(SORTED_FEATURE_NAMES_JSON_PATH[exp_id], list(mitten_pvalues.keys()))
        _single_feature_visualization(mitten_feature, mitten_pvalues)
        plot_feature.plot_feature_hierarchy_structure(
            mitten_feature, save_name=path.join(_tmp_save_name, mitten_feature.name,
                                                mitten_feature.ref_img.exp_id + "_hierarchy"),
            feature_names=mitten_feature_names,
            sorted_p_value_dict=mitten_pvalues,
        )

        plot_feature.plot_feature_distribution_calb2(
            mitten_feature, save_name=path.join(_tmp_save_name, mitten_feature.name,
                                                mitten_feature.ref_img.exp_id + "_features_by_periods.png"),
            sorted_p_value_dict=mitten_pvalues, period_name_flag=True
        )
        plot_feature.plot_feature_distribution_calb2(
            mitten_feature, save_name=path.join(_tmp_save_name, mitten_feature.name,
                                                mitten_feature.ref_img.exp_id + "_features_by_pvalues.png"),
            sorted_p_value_dict=mitten_pvalues
        )


