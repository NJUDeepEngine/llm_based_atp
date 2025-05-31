import math
from enum import Enum
from abc import ABC, abstractmethod
from functools import total_ordering
from dataclasses import dataclass, field
from typing import Optional, List, Iterable, Union, Set

from dojo import (
    TacticState,
    ProofFinished,
    LeanError,
    ProofGivenUp,
)

class Status(Enum):
    """
    Status of a node in the search tree.
    """
    SOLVED = 0
    UNSOLVED = 1
    INVALID = 2

class Node(ABC):
    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def distance_to_proof(self) -> int:
        "The smallest number of steps to a proof."
        raise NotImplementedError
    
    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @property
    def id(self):
        return self.leandojo_state.id
    
@dataclass
class SolvedNode(Node):
    """
    A node in the search tree that has been solved.
    """
    leandojo_state: ProofFinished
    in_edge: Optional['Edge'] = None 
    status: Status = Status.SOLVED
    distance_to_proof: float = 0.0
    is_terminal: bool = True
    depth: Optional[int] = None

    

@dataclass
class InvalidNode(Node):
    """
    A node in the search tree that is invalid.
    """
    leandojo_state: Union[LeanError, ProofGivenUp]
    in_edge: Optional['Edge'] = None 
    status: Status = Status.INVALID
    distance_to_proof: float = math.inf
    is_terminal: bool = True
    depth: Optional[int] = None

    

@total_ordering
@dataclass(unsafe_hash=True)
class UnsolvedNode(Node):
    """
    A node in the search tree that has not been solved.
    """
    leandojo_state: TacticState = field(compare=True)
    status: Status = field(default=Status.UNSOLVED, init=False, compare=False, repr=True)
    is_terminal: bool = False
    priority: float = field(default=0.0, compare=False)
    depth: Optional[int] = field(default=None, compare=False, repr=False)

    in_edges: List['Edge'] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    _out_edges: Optional[List['Edge']] = field(
        default=None, init=False, compare=False, repr=False
    )

    _distance_to_proof: float = field(default=math.inf, init=False, compare=False, repr=False)

    success_edges_list =[]

    @property
    def distance_to_proof(self) -> float:
        return self._distance_to_proof

    @property
    def out_edges(self) -> Optional[List['Edge']]:
        return self._out_edges
    
    @out_edges.setter
    def out_edges(self, edges: Iterable['Edge']) -> None:
        self._out_edges = list(edges)
        self._recompute_status()
        self._recompute_distance_to_proof()

    def _recompute_status(self):
        assert self.out_edges is not None
        # recompute current node status
        if self.status != Status.UNSOLVED:
            return
        if any(edge.dst.status == Status.SOLVED for edge in self.out_edges):
            self.status = Status.SOLVED
        if all(edge.dst.status == Status.INVALID for edge in self.out_edges):
            self.status = Status.INVALID

        # update the status of the parent node
        if self.status != Status.UNSOLVED:
            for edge in self.in_edges:
                edge.src._recompute_status()

    def _recompute_distance_to_proof(self):
        if self.out_edges:
            distance = min(edge.distance_to_proof() for edge in self.out_edges)
        else:
            distance = math.inf

        if distance < self._distance_to_proof:
            self._distance_to_proof = distance
            for edge in self.in_edges:
                edge.src._recompute_distance_to_proof()


    def __lt__(self, other: "UnsolvedNode") -> bool:
        return self.priority < other.priority

    def extract_proof(self) -> Optional[List["Edge"]]:
        """
        Extract a proof of the current node as a sequence of edges.
        """
        if self.status != Status.SOLVED:
            return None
        
        # return []

        proving_path = min(
            self.out_edges,
            key=Edge.distance_to_proof
        )

        if proving_path.dst.is_terminal:
            assert isinstance(proving_path.dst, SolvedNode)
            return [proving_path]
        else:
            assert isinstance(proving_path.dst, UnsolvedNode)
            remaining_proof = proving_path.dst.extract_proof()
            return [proving_path] + remaining_proof
    
    def is_descendant(self, potential_descendant: Node) -> bool:
        assert isinstance(potential_descendant, UnsolvedNode), f"potential_descendant should be UnsolvedNode, got {type(potential_descendant)}"
        if self == potential_descendant:
            return True
        if isinstance(self, UnsolvedNode) and self.out_edges:
            for edge in self.out_edges:
                if isinstance(edge.dst, UnsolvedNode) and edge.dst.is_descendant(potential_descendant):
                    return True
        return False
    


@dataclass
class Edge:
    """
    An edge in the search tree, representing a tactic.
    """
    src: Node = field(repr=False)
    dst: Node = field(repr=False)
    tactic: str
    score: float

    def distance_to_proof(self) -> float:
        return 1 + self.dst.distance_to_proof
    
    def __repr__(self) -> str:
        src_id = getattr(self.src, 'id', 'unknown')
        dst_id = getattr(self.dst, 'id', 'unknown')
        return f"Edge({src_id} -> {dst_id})"
    

def print_search_tree(root: Node) -> str:
    lines = []
    visited = set()

    def dfs(node: Node, depth: int):
        indent = "  " * depth
        node_id = id(node)

        if node_id in visited:
            lines.append(f"{indent}🔁 Node (revisited)")
            return
        visited.add(node_id)

        # 打印当前节点
        if isinstance(node, UnsolvedNode):
            lines.append(f"{indent}📍 UnsolvedNode: {str(node.id)}")
            if node.out_edges:
                for edge in node.out_edges:
                    tactic_str = edge.tactic
                    lines.append(f"{indent}  ├─🔧 Tactic: {tactic_str}")
                    dfs(edge.dst, depth + 2)
        elif isinstance(node, SolvedNode):
            lines.append(f"{indent}✅ SolvedNode:{str(node.id)}")
        elif isinstance(node, InvalidNode):
            lines.append(f"{indent}❌ InvalidNode: {str(node.id)}")
        else:
            lines.append(f"{indent}❓ Unknown Node Type")

    dfs(root, 0)
    return "\n".join(lines)




def collect_success_edges(root_node: Node) -> List['Edge']:
    success_edges: List['Edge'] = []
    edge_path: List['Edge'] = []

    def dfs(node: Node):
        if isinstance(node, SolvedNode):
            for edge in edge_path:
                if edge not in success_edges:
                    success_edges.append(edge)
            return
        if isinstance(node, UnsolvedNode) and node.out_edges:
            for edge in node.out_edges:
                edge_path.append(edge)
                dfs(edge.dst)
                edge_path.pop()

    dfs(root_node)
    return success_edges
