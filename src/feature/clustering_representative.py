import numpy as np
from typing import List, Callable, Optional, Dict, Union, Tuple
from dataclasses import dataclass, field, MISSING
import os
import os.path as path

from src.config import *
from src.basic.utils import *
from src.data_manager import *
from src.feature.feature_utils import *
from src.feature.feature_manager import *
from src.feature.clustering import *
from src.feature.clustering_params import *
from src.basic.data_operator import *


def find_representative_clustering(
        vec_space: VectorSpace,
        num_embeddings: int = 10,
        num_labelling: int = 5,) -> Dict[int, List[List[Embedding]]]:
    # find the threshold of score
    candidate_labellings = []
    for single_labelling, embed_list in vec_space.all_embed_by_labelling.items():
        avg_score = np.mean([single_embed.score for single_embed in embed_list])
        occurrences = len(embed_list)
        if single_labelling.n_cluster in PLOTTING_CLUSTERS_OPTIONS:
            candidate_labellings.append({
                "labelling": single_labelling,
                "embed_list": embed_list,
                "avg_score": avg_score,
                "n_cluster": single_labelling.n_cluster,
                "occurrences": occurrences
            })
    # all_candidate_scores = [item["avg_score"] for item in candidate_labellings]
    # score_threshold = np.percentile(all_candidate_scores, q=TOP_LABELLING_THRESHOLD)
    # all_candidate_occurrences = [item["occurrences"] for item in candidate_labellings]
    # occurrence_threshold = np.percentile(all_candidate_occurrences, q=TOP_LABELLING_OCCURRENCE_THRESHOLD)

    # find the best embedding with above threshold score and largest occurrence
    best_labellings_by_cluster = defaultdict(list)
    for item in candidate_labellings:
        if item["occurrences"] > TOP_LABELLING_OCCURRENCE_THRESHOLD:
            current_n_cluster = item["n_cluster"]
            current_score = item["avg_score"]
            # Check if this cluster is not yet in results or if this score is better
            best_embedding_list = list(sorted(item["embed_list"], key=lambda embed: embed.score, reverse=True))
            relabelled_embedding_list = [relabel_sorting(single_embed, vec_space)
                                         for single_embed in best_embedding_list[:num_embeddings]]

            best_labellings_by_cluster[current_n_cluster].append((relabelled_embedding_list, current_score))
    best_embedding_by_n_cluster = {
        n_cluster: list(sorted(best_labellings_by_cluster[n_cluster], key=lambda x: x[1], reverse=True))[:num_labelling]
        for n_cluster in PLOTTING_CLUSTERS_OPTIONS if n_cluster in best_labellings_by_cluster}
    return best_embedding_by_n_cluster


def relabel_sorting(single_embed: Embedding, vec_space: VectorSpace) -> Embedding:
    feature_zero_by_label = defaultdict(list)
    for cell_uid, single_label in single_embed.label_by_cell.items():
        feature_zero_by_label[int(single_label)].append(vec_space.by_cells[cell_uid][2])
    avg_feature_zero = average_dict(feature_zero_by_label)
    sorted_old_labels = sorted(avg_feature_zero, key=avg_feature_zero.get)
    old_to_new_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_old_labels)}
    new_labels = [old_to_new_mapping[single_label] for single_label in single_embed.labels]
    return Embedding(name=single_embed.name + "_relabel", n_dim=single_embed.n_dim, cells_uid=single_embed.cells_uid,
                     embedding=single_embed.embedding, params=single_embed.params, score=single_embed.score,
                     labels=np.array(new_labels))


def load_representative_clustering(vec_space: VectorSpace, overwrite=False,) -> Dict[EmbedUID, Embedding]:
    if overwrite or (not path.exists(vec_space.representative_embeddings_path)):
        vec_space.prepare_embedding()
        representative_clustering_by_n_cluster = find_representative_clustering(vec_space)
        print(f"Saving representative embeddings into {vec_space.representative_embeddings_path}")
        os.makedirs(path.dirname(vec_space.representative_embeddings_path), exist_ok=True)
        with pd.ExcelWriter(vec_space.representative_embeddings_path, engine='xlsxwriter') as writer:
            for n_cluster, labelling_list in representative_clustering_by_n_cluster.items():
                for labelling_id, (embed_list, labelling_score) in enumerate(labelling_list):
                    for embed_id, single_embed in enumerate(embed_list):
                        vec_space.save_single_embedding(
                            single_embed, writer,
                            new_sheet_name=f"c{n_cluster} l{labelling_id} e{embed_id}")
        print(f"Embedding sheets saved at: {vec_space.representative_embeddings_path}.")

    print(f"Loading embeddings from {vec_space.representative_embeddings_path}...")
    xls = pd.ExcelFile(vec_space.representative_embeddings_path, engine='openpyxl')
    representative_clustering_by_embed_uid = {}
    for sheet_name in xls.sheet_names:
        c_string, l_string, e_string = sheet_name.split(" ")
        assert c_string[0] == 'c' and l_string[0] == 'l' and e_string[0] == 'e'
        embed_uid = EmbedUID(n_cluster=int(c_string[1:]), labelling_id=int(l_string[1:]), embed_id=int(e_string[1:]))
        new_embedding = vec_space.load_single_embedding(xls, sheet_name)
        representative_clustering_by_embed_uid[embed_uid] = new_embedding
    print("Loading complete.")
    return representative_clustering_by_embed_uid
