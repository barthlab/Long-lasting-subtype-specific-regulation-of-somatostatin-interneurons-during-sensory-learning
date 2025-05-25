import matplotlib.pyplot as plt

from src import *


def _clustering_examples_visualization(vec_space: VectorSpace, feature_db: FeatureDataBase):
    vec_space.prepare_embedding()
    print([(single_embed.params, single_embed.score)
           for single_embed in vec_space.get_embeddings(top_k=200)])
    _tmp_save_name = path.join("6_main_figure", feature_db.ref_img.exp_id + f"_{vec_space.name}_summary.png")
    plot_clustering.plot_embedding_summary(vec_space, save_name=_tmp_save_name)


def _labelling_occurrence_labeling(vec_space: VectorSpace):
    _tmp_save_name = path.join("6_main_figure",
                               vec_space.ref_feature_db.exp_id + f"_{vec_space.name}_occurrence.png")
    vec_space.prepare_embedding()
    plot_clustering_evaluate.plot_labeling_quality_summary(
        vec_space, save_name=_tmp_save_name, embed_dict=vec_space.all_embed_by_labelling)


def _embed_quality_visualize(vec_space: VectorSpace, representative_clustering_by_embed_uid: Dict[EmbedUID, Embedding],
                             mini_flag=False):
    _root_save_name = path.join("6_main_figure", vec_space.ref_feature_db.exp_id,)

    for embed_uid, best_embed in representative_clustering_by_embed_uid.items():
        if mini_flag:
            _tmp_save_name = path.join(_root_save_name, "embed_quality", f"{embed_uid.n_cluster}clusters",
                                       f"labelling{embed_uid.labelling_id}")
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{embed_uid.in_short()}_cell_type.png",
                cell_type_flag=True, ellipse_flag=False, size=(2.5, 2.5))
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{embed_uid.in_short()}_cluster_id.png",
                cell_type_flag=False, ellipse_flag=False, size=(2.5, 2.5))
        else:
            _tmp_save_name = path.join(_root_save_name,)
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{embed_uid.in_short()}_cell_type.png",
                cell_type_flag=True, ellipse_flag=False, size=(2.5, 2.5))
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{embed_uid.in_short()}_cell_type_small.png",
                cell_type_flag=True, ellipse_flag=False, size=(2, 2))
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{embed_uid.in_short()}_cell_type_ellipse.png",
                cell_type_flag=True, ellipse_flag=True, size=(2, 2), label_flag=False)
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{embed_uid.in_short()}_cluster_id.png",
                cell_type_flag=False, ellipse_flag=False, size=(2.5, 2.5))
            plot_clustering.plot_one_beautiful_embedding(
                vec_space, best_embed, save_name=_tmp_save_name + f"_best_embed_{embed_uid.in_short()}_cluster_id_small.png",
                cell_type_flag=False, ellipse_flag=False, size=(2, 2))


def _plasticity_complex_visualize(feature_db: FeatureDataBase, vis_feature_names: List[str],
                                  representative_clustering_by_embed_uid: Dict[EmbedUID, Embedding]):
    _tmp_save_name = path.join("6_main_figure", feature_db.ref_img.exp_id)
    basic_evoked_response_and_fold_change(feature_db)
    cell_types = reverse_dict(feature_db.cell_types)
    if not feature_db.Ai148_flag:
        for bar_name, (bar_config, sort_name) in PLOTTING_GROUPS_CONFIG[feature_db.exp_id].items():
            plot_clustering_evaluate.plot_fold_change_bars(
                path.join(_tmp_save_name, f"fold_change_bars_GT_{bar_name}.png"),
                feature_db, bars_list=bar_config,
                group_of_cell_list={
                    CELLTYPE2STR[CellType.Calb2_Neg]: cell_types[CellType.Calb2_Neg],
                    CELLTYPE2STR[CellType.Calb2_Pos]: cell_types[CellType.Calb2_Pos],
                },
                group_colors=[
                    CELLTYPE2COLOR[CellType.Calb2_Neg],
                    CELLTYPE2COLOR[CellType.Calb2_Pos],
                ], size=(2.5, 1.5)
            )
    for embed_uid, representative_embedding in representative_clustering_by_embed_uid.items():
        n_cluster = embed_uid.n_cluster
        cluster_id_dict = reverse_dict(representative_embedding.label_by_cell)
        renamed_cluster_id_dict, cluster_group_color = {}, []
        for cluster_name, cluster_id in zip((1, 2, 3), (0, 2, 1)):
            renamed_cluster_id_dict[f"{cluster_name}"] = cluster_id_dict[cluster_id]
            cluster_group_color.append(CLUSTER_COLORLIST[cluster_id])
        if not feature_db.Ai148_flag:
            renamed_cluster_id_dict = {
                CELLTYPE2STR[CellType.Calb2_Neg]: cell_types[CellType.Calb2_Neg],
                CELLTYPE2STR[CellType.Calb2_Pos]: cell_types[CellType.Calb2_Pos],
                **renamed_cluster_id_dict
            }
            cluster_group_color = [
                CELLTYPE2COLOR[CellType.Calb2_Neg],
                CELLTYPE2COLOR[CellType.Calb2_Pos],
                *cluster_group_color
            ]
        plot_clustering_evaluate.plot_feature_summary(
            path.join(_tmp_save_name, f"feature_summary_{embed_uid.in_short()}.png"),
            feature_db, vis_feature_names, selected_days=FEATURE_SELECTED_DAYS,
            group_of_cell_list=renamed_cluster_id_dict,
            group_colors=cluster_group_color, size=(0.7/1.5, 0.3)
        )
        if n_cluster == BEST_NUM_CLUSTERS:
            for bar_name, (bar_config, sort_name) in PLOTTING_GROUPS_CONFIG[feature_db.exp_id].items():
                plot_clustering_evaluate.plot_fold_change_bars(
                    path.join(_tmp_save_name, f"fold_change_bars_{embed_uid.in_short()}_major_{bar_name}.png"),
                    feature_db, bars_list=bar_config,
                    group_of_cell_list={
                        CELLTYPE2STR[CellType.Put_Calb2_Neg]:
                            list(chain.from_iterable([cluster_id_dict[i] for i in range(1, n_cluster)])),
                        CELLTYPE2STR[CellType.Put_Calb2_Pos]: cluster_id_dict[0],
                    },
                    group_colors=[
                        CELLTYPE2COLOR[CellType.Put_Calb2_Neg],
                        CELLTYPE2COLOR[CellType.Put_Calb2_Pos],
                    ], size=(2.5, 1.5)
                )
                plot_clustering_evaluate.plot_fold_change_bars(
                    path.join(_tmp_save_name, f"fold_change_bars_{embed_uid.in_short()}_{bar_name}.png"),
                    feature_db, bars_list=bar_config,
                    group_of_cell_list={f"cluster {int(i+1)}": cluster_id_dict[i] for i in reversed(range(n_cluster))},
                    group_colors=[CLUSTER_COLORLIST[i] for i in reversed(range(n_cluster))], size=(2.5, 1.5)
                )


def _heatmap_by_clusters(single_image: Image, feature_db: FeatureDataBase, representative_embedding: Embedding):
    cluster_id_dict = reverse_dict(representative_embedding.label_by_cell)
    for trials_name, trials_criteria in PLOTTING_TRIALS_CONFIG.items():
        for col_name, (col_criteria, sort_day) in PLOTTING_DAYS_CONFIG[feature_db.exp_id].items():
            _tmp_save_name = path.join("6_main_figure", f"{single_image.exp_id}",
                                       f"{single_image.exp_id}_{trials_name}_{col_name}")
            days_dict = {col_name: feature_db.days_ref[col_name] for col_name in col_criteria}
            for cluster_id, cells_uid_list in cluster_id_dict.items():
                if not feature_db.Ai148_flag:
                    plot_overview.plot_heatmap_overview_cellwise(
                        single_image.select(Calb2=1, cell_uid=tuple(cells_uid_list)),
                        feature_db, save_name=_tmp_save_name + f"_cluster{cluster_id}_Calb2_pos.png",
                        days_dict=days_dict, trials_criteria=trials_criteria,
                        sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Calb2_Pos],
                        zscore_flag=True
                    )
                    plot_overview.plot_heatmap_overview_cellwise(
                        single_image.select(Calb2=0, cell_uid=tuple(cells_uid_list)),
                        feature_db, save_name=_tmp_save_name + f"_cluster{cluster_id}_Calb2_neg.png",
                        days_dict=days_dict, trials_criteria=trials_criteria,
                        sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Calb2_Neg],
                        zscore_flag=True
                    )
                plot_overview.plot_heatmap_overview_cellwise(
                    single_image.select(cell_uid=tuple(cells_uid_list)),
                    feature_db, save_name=_tmp_save_name + f"_cluster{cluster_id}_all_cells.png",
                    days_dict=days_dict, trials_criteria=trials_criteria,
                    sorting=(sort_day, brightness), theme_color=CELLTYPE2COLOR[CellType.Unknown],
                    zscore_flag=True
                )


if __name__ == "__main__":
    all_feature_names = json_load(SORTED_FEATURE_NAMES_JSON_PATH["Calb2_SAT"])

    for top_k_num in (30, ):
        topk_mitten_features = all_feature_names[:top_k_num]
        print(len(topk_mitten_features), topk_mitten_features)

        for exp_id in ("Calb2_SAT", "Ai148_SAT", ):  #"Ai148_PSE", "Calb2_PSE"):
            mitten = Experiment(exp_id=exp_id)
            mitten_data = mitten.image
            mitten_feature = FeatureDataBase("mitten_feature", mitten_data)
            feature_prepare(mitten_feature)
            mitten_vector = get_feature_vector(mitten_feature, topk_mitten_features,
                                               FEATURE_SELECTED_DAYS, f"Top{top_k_num}_ACC456")
            mitten_representative = load_representative_clustering(mitten_vector)

            if DEBUG_FLAG:
                _embed_quality_visualize(mitten_vector, mitten_representative, mini_flag=True)
                continue
            else:
                selected_embeddings = general_select(mitten_representative, **CHOSEN_ONE[exp_id])
                # _clustering_examples_visualization(mitten_vector, mitten_feature)
                # _labelling_occurrence_labeling(mitten_vector)
                _embed_quality_visualize(mitten_vector, selected_embeddings)
                # _heatmap_by_clusters(mitten_data, mitten_feature, selected_embeddings[EmbedUID(**CHOSEN_ONE[exp_id])])
                _plasticity_complex_visualize(mitten_feature, vis_feature_names=topk_mitten_features,
                                              representative_clustering_by_embed_uid=selected_embeddings)

