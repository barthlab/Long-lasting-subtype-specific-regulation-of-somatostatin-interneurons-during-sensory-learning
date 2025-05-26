from src import *


def _every_fov(single_exp: Experiment, feature_db: FeatureDataBase, single_embed: Embedding):
    for single_mice in single_exp.mice:
        for single_fov in single_mice.fovs:
            for valid_cell_session in general_filter(single_fov.cell_sessions,
                                                     day_id=lambda x: x in feature_db.days_ref["ACC456"]):
                post_fix = f"{valid_cell_session.cell_id}_{valid_cell_session.day_id.name}"
                plot_example.plot_image_example(
                    Image(single_exp.exp_id, [valid_cell_session,]), feature_db, single_embed,
                    path.join("7_examples", single_exp.exp_id, "every_fov",
                              f"{single_fov.str_uid}_{post_fix}_cluster_id.png"),
                    color_by_cell_type_flag=False)
                plot_example.plot_image_example(
                    Image(single_exp.exp_id, [valid_cell_session,]), feature_db, single_embed,
                    path.join("7_examples", single_exp.exp_id, "every_fov",
                              f"{single_fov.str_uid}_{post_fix}_cell_type.png"),
                    color_by_cell_type_flag=True)


def _representative_feature(
        feature_db: FeatureDataBase, single_embed: Embedding
):
    n_cluster = single_embed.n_cluster
    cluster_id_dict = reverse_dict(single_embed.label_by_cell)
    renamed_cluster_id_dict, cluster_group_color = {}, []
    for cluster_name, cluster_id in zip((1, 2, 3), (0, 2, 1)):
        renamed_cluster_id_dict[f"{cluster_name}"] = cluster_id_dict[cluster_id]
        cluster_group_color.append(CLUSTER_COLORLIST[cluster_id])
    if not feature_db.Ai148_flag:
        vis_feature_names = [
            'response prob || evoked-trial period || stimulus trial only || 2std',
            # 'amplitude || inter-trial blocks || 1std',
            'peak || evoked-trial period || all trial included',
        ]
    else:
        vis_feature_names = [
            'response prob || evoked-trial period || stimulus trial only || 5std',
            # 'amplitude || inter-trial blocks || 2std',
            'peak || evoked-trial period || all trial included',
        ]
    plot_example.plot_feature_example(
        path.join("7_examples", feature_db.exp_id, f"feature_summary.png"),
        feature_db, vis_feature_names,
        selected_days=FEATURE_SELECTED_DAYS,
        group_of_cell_list=renamed_cluster_id_dict,
        group_colors=cluster_group_color, size=(1.2, 1.2)
    )

    if not feature_db.Ai148_flag:
        cell_types = reverse_dict(feature_db.cell_types)
        renamed_cluster_id_dict = {
            CELLTYPE2STR[CellType.Calb2_Pos]: cell_types[CellType.Calb2_Pos],
            CELLTYPE2STR[CellType.Calb2_Neg]: cell_types[CellType.Calb2_Neg],
        }
        cluster_group_color = [
            CELLTYPE2COLOR[CellType.Calb2_Pos],
            CELLTYPE2COLOR[CellType.Calb2_Neg],
        ]
        plot_example.plot_feature_example(
            path.join("7_examples", feature_db.exp_id, f"feature_summary_calb2.png"),
            feature_db, vis_feature_names,
            selected_days=FEATURE_SELECTED_DAYS,
            group_of_cell_list=renamed_cluster_id_dict,
            group_colors=cluster_group_color, size=(1.2, 1.2)
        )


if __name__ == "__main__":
    all_feature_names = json_load(SORTED_FEATURE_NAMES_JSON_PATH["Calb2_SAT"])
    top30_mitten_features = all_feature_names[:30]
    print(len(top30_mitten_features), top30_mitten_features)
    # for exp_id in ("Calb2_SAT", "Ai148_SAT",):
    for exp_id in ("Ai148_PSE", "Calb2_PSE",):
        mitten = Experiment(exp_id=exp_id)
        mitten_data = mitten.image
        mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
        feature_prepare(mitten_feature)
        mitten_vector = get_feature_vector(mitten_feature, top30_mitten_features,
                                           FEATURE_SELECTED_DAYS, f"Top30_ACC456")
        mitten_representative = load_representative_clustering(mitten_vector)

        selected_embeddings = general_select(mitten_representative, **CHOSEN_ONE[exp_id])
        for mitten_representative_embedding in selected_embeddings.values():
            reassign_label_for_visualization(mitten_representative_embedding, mitten_feature)
        _every_fov(mitten, mitten_feature, selected_embeddings[EmbedUID(**CHOSEN_ONE[exp_id])])
        _representative_feature(mitten_feature, selected_embeddings[EmbedUID(**CHOSEN_ONE[exp_id])])


