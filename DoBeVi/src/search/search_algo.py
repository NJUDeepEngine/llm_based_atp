import os
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import asyncio
import ray
from ray.util.queue import Queue
from ray.exceptions import RayTaskError

from dojo import *
from search.search_tree import (
    Status,
    SolvedNode,
    UnsolvedNode,
    InvalidNode,
    Edge,
)
from search.tactic_generator import (
    TacticGenerator,
    HuggingFaceGenerator,
    VllmGenerator,
    InternlmVllmGenerator,
)


@dataclass(frozen=True)
class SearchResult:
    """
    The result of a search.
    """
    theorem: TracedTheorem
    status: Status
    proof: Optional[List[str]]
    num_total_nodes: int
    num_expansions: int
    elapsed_time: float
    dojo_elapsed_time: Optional[float]
    model_elapsed_time: float

    def to_dict(self) -> dict:
        theorem = dict(
            url=str(self.theorem.root_dir),
            file_path=str(self.theorem.path),
            name=self.theorem.name,
        )
        result = dict(
            theorem=theorem,
            status=self.status.name,
            proof=self.proof if self.proof else [],
            num_total_nodes=self.num_total_nodes,
            num_expansions=self.num_expansions,
            elapsed_time=self.elapsed_time,
            dojo_elapsed_time=self.dojo_elapsed_time,
            model_elapsed_time=self.model_elapsed_time,
        )
        return result
    
class Prover(ABC):
    def __init__(
        self,
        tactic_generators: List[TacticGenerator],
        prover_id: int,
        search_timeout: int,
        max_expansions: Optional[int],
        num_sampled_tactics: int,
    ):
        self.tactic_generators = tactic_generators
        self.prover_id = prover_id
        self.search_timeout = search_timeout
        self.max_expansions = max_expansions
        self.num_sampled_tactics = num_sampled_tactics

    @abstractmethod
    async def search(self, thm: TracedTheorem) -> Optional[SearchResult]:
        """
        Search for a proof of a theorem.
        """
        raise NotImplementedError
    
    def select_tac_gen(self) -> TacticGenerator:
        # TODO: implement polling strategy in the future
        return self.tactic_generators[self.prover_id % len(self.tactic_generators)] 

@ray.remote
class ProverActor:
    def __init__(
        self, 
        clazz: Prover,
        actor_id: int,
        tactic_generators: List[TacticGenerator],
        max_expansions: Optional[int],
        num_sampled_tactics: int,
        search_timeout: int,
        queue_timeout: int,
    ):
        self.actor_id = actor_id
        self.tactic_generators = tactic_generators
        self.max_expansions = max_expansions
        self.search_timeout = search_timeout
        self.queue_timeout = queue_timeout
        self.prover = clazz(
            tactic_generators=tactic_generators,
            actor_id=actor_id,
            search_timeout=search_timeout,
            max_expansions=max_expansions,
            num_sampled_tactics=num_sampled_tactics,
        )

    async def search(
        self,
        input_queue: Queue,
        output_queue: Queue,
    ) -> None:
        while True:
            try:
                thm = input_queue.get(timeout=self.queue_timeout)
            except Exception:
                continue
            
            if thm is None:
                break

            try:
                result = await asyncio.wait_for(
                    self.prover.search(thm),
                    timeout=self.search_timeout,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                logging.error(
                    f"🚨[Actor {self.actor_id}] search({thm.name!r}) timed-out "
                    f"after {self.search_timeout}s"
                )
                result = None
            except Exception:
                logging.error(
                    f"🚨[Actor {self.actor_id}] unexpected error in search({thm.name!r})"
                )
                result = None
            
            try:
                output_queue.put(result)
            except Exception:
                raise
            
class ProverScheduler:
    def __init__(
        self, 
        model_path: str,
        num_workers: int,
        num_gpus: int,
        prover_clazz: Prover,
        search_timeout: int,
        max_expansions: Optional[int],
        num_sampled_tactics: int,
    ):
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.search_timeout = search_timeout
        self.results = []

        # init tactic generators and prover actors
        RAY_TEMP_DIR = os.getenv("RAY_TEMP_DIR", None)
        if RAY_TEMP_DIR:
            ray.init(_temp_dir=RAY_TEMP_DIR)
        else:
            ray.init()
            
        self.tactic_generators = [
            VllmGenerator.remote(
                model_path=model_path,
                length_penalty=1.0,
                max_length=4096,
                gpu_id=i % self.num_gpus,
            )
            for i in range(self.num_gpus)
        ]
        
        # self.tactic_generators = [
        #     InternlmVllmGenerator.remote(
        #         model_path=tac_gen_path,
        #         length_penalty=1.0,
        #         max_length=4096,
        #         gpu_id=i % self.num_gpus,
        #     )
        #     for i in range(self.num_gpus)
        # ]

        self.prover_actors = [
            ProverActor.remote(
                clazz=prover_clazz,
                actor_id=i,
                tactic_generators=self.tactic_generators,
                max_expansions=max_expansions,
                num_sampled_tactics=num_sampled_tactics,
                search_timeout=search_timeout,
                queue_timeout=5,
            )
            for i in range(num_workers)
        ]
    
    def search(
        self,
        theorems: List[TracedTheorem],
    ) -> List[Optional[SearchResult]]:
        input_queue = Queue()
        output_queue = Queue()

        for thm in theorems:
            input_queue.put(thm)
        
        for _ in range(self.num_workers):
            input_queue.put(None)
        
        # start prover actors
        worker_tasks = [
            self.prover_actors[i].search.remote(input_queue, output_queue) 
            for i in range(self.num_workers)
        ]

        # monitor the progress
        pbar = tqdm(total=len(theorems))
        async def monitor():
            while len(self.results) < len(theorems):
                try:
                    result = output_queue.get()
                except Exception as e:
                    logging.error(f"🚨{type(e).__name__}: {e}")
                    result = None
                self.results.append(result)
                pbar.update(1)
            pbar.close()
        asyncio.run(monitor())

        # stop prover actors
        ray.get(worker_tasks)
        # ray.shutdown()
        
        return self.results