import json
import os
from typing import List, Tuple, Dict, Optional
from search.search_algo import SearchResult, Status
from search.best_first_search_algo import BestFirstSearchProver
from search.group_score_search_algo import GroupScoreSearchProver
from search.internlm_bfs_algo import InternlmBFSProver
from search.layer_dropout_algo import LayerDropoutProver

def get_num_gpus(cuda_visible_devices: str) -> int:
    return len(cuda_visible_devices.split(","))

def get_prover_clazz(algo_str: str):
    if algo_str == "best_first":
        return BestFirstSearchProver
    elif algo_str == "group_score":
        return GroupScoreSearchProver
    elif algo_str == "internlm_bfs":
        return InternlmBFSProver
    elif algo_str == "layer_dropout":
        return LayerDropoutProver
    else:
        raise ValueError(f"Invalid algorithm: {algo_str}")

def _get_proof_length(results: List[Optional[SearchResult]]) -> Dict:
    path_len_dict = {}
    for result in results:
        if result == None or result.status != Status.SOLVED:
            continue
        path_len = len(result.proof)
        path_len_dict[path_len] = path_len_dict.get(path_len, 0) + 1
    return path_len_dict
    
def get_stats(results: List[Optional[SearchResult]]) -> Dict:
    stats_dict = {}
    stats_dict['proof_length'] = _get_proof_length(results)
    return stats_dict


