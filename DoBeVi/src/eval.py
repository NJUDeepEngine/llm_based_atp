import os
os.environ['RAY_TEMP_DIR'] = '/path/to/ray_temp'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['RAY_DEDUP_LOGS'] = '0' 
os.environ['RAY_memory_monitor_refresh_ms'] = '0'
os.environ["VLLM_CONFIGURE_LOGGING"] = '0'

import logging
from typing import Optional, List, Tuple
import json
from pathlib import Path
import datetime
from datetime import timedelta
import time

from config import settings
from dojo import TracedRepo
from search.search_tree import Status
from search.search_algo import ProverScheduler, SearchResult
from search.best_first_search_algo import BestFirstSearchProver

from utils import (
    get_num_gpus,
    get_prover_clazz,
    get_stats,
)

def evaluate(
    repo_path: str,
    file_path: str,
    model_path: str,
    algorithm: str,
    num_workers: int,
    num_gpus: int,
    num_sampled_tactics: int,
    search_timeout: int,
    max_expansions: Optional[int],
    result_save_path: bool,
) -> float:
    # generate theorems
    repo = TracedRepo(repo_path)
    theorems_dict = repo.get_traced_theorems_from_file(file_path, False)
    theorems = []
    for thm_name, thms in theorems_dict.items():
        theorems.extend(thms)
    
    # create prover scheduler and search
    scheduler = ProverScheduler(
        model_path=model_path,
        num_workers=num_workers,
        num_gpus=get_num_gpus(os.getenv("CUDA_VISIBLE_DEVICES", "0")) if num_gpus == 0 else num_gpus,
        prover_clazz=get_prover_clazz(algorithm),
        search_timeout=search_timeout,
        max_expansions=max_expansions,
        num_sampled_tactics=num_sampled_tactics,
    )

    time_start = time.time()
    results = scheduler.search(theorems)
    time_end = time.time()

    # evaluate the results
    num_solved = num_failed = num_discarded = 0
    for result in results:
        if result is None:
            num_discarded += 1
        elif result.status == Status.SOLVED:
            num_solved += 1
        else:
            num_failed += 1

    if num_solved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_solved / (num_solved + num_failed)

    os.makedirs(result_save_path, exist_ok=True)
    result_file = os.path.join(result_save_path, f"{Path(file_path).stem}_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, "w") as f:
        result_dict = {}
        result_dict['pass_1'] = pass_1
        result_dict['pass_num'] = num_solved
        result_dict['args'] = {
            'repo_path': repo_path,
            'file_path': file_path,
            'algorithm': algorithm,
            'model_path': model_path,
            'num_workers': num_workers,
            'num_gpus': num_gpus,
            'num_sampled_tactics': num_sampled_tactics,
            'search_timeout': search_timeout,
            'max_expansions': max_expansions,
            'result_save_path': result_save_path,
        }
        result_dict['total_time'] = str(timedelta(seconds=time_end - time_start))
        result_dict['stats'] = get_stats(results)
        result_dict['results'] = [r.to_dict() if r is not None else None for r in results]
        logging.info(result_dict)
        json.dump(result_dict, f, indent=4)

    logging.info(f"Pass@1: {pass_1}")
    logging.info(f"Pass Num: {num_solved}")
    return pass_1

def main() -> None:
    pass_1 = evaluate(
        settings.REPO_PATH,
        settings.FILE_PATH,
        settings.MODEL_PATH,
        settings.ALGORITHM,
        settings.NUM_WORKERS,
        settings.NUM_GPUS,
        settings.NUM_SAMPLED_TACTICS,
        settings.SEARCH_TIMEOUT,
        settings.MAX_EXPANSIONS,
        settings.RESULT_SAVE_PATH
    )
    logging.info(f"Pass@1: {pass_1}")

if __name__ == '__main__':
    main()