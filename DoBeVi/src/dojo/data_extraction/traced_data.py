import os
import json
import shutil
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field

from .ast import (
    Node,
    FileNode,
    LemmaNode,
    ModulePreludeNode,
    CommandTheoremNode,
    CommandModuledocNode,
    CommandDoccommentNode,
    MathlibTacticLemmaNode,
    is_leaf,
)
from typing import List, Dict, Tuple, Union
from .lean_data import LeanFile, LeanRepo, LeanTheorem, Pos
from ..utils import working_directory, execute
from ..constants import (
    _COMMENT_REGEX, 
    LEAN4_PACKAGES_DIR, 
    LEAN4_REPL_PATH, 
    LEAN4_BUILD_DIR,
    LEAN4_DATA_EXTRACTOR_PATH,
)
from ..errors import RepoInitError, DojoInitError

@dataclass(frozen=True)  
class Comment:
    """A comment in a Lean file."""

    start: Pos
    end: Pos
    text: str

    def __post_init__(self) -> None:
        assert isinstance(self.start, Pos)
        assert isinstance(self.end, Pos)
        assert self.start <= self.end
        assert isinstance(self.text, str)

def _collect_lean4_comments(ast: FileNode) -> List[Comment]:
    comments = []

    def _callback(node, _):
        nonlocal comments
        if isinstance(node, CommandModuledocNode) or isinstance(node, CommandDoccommentNode):
            comments.append(Comment(node.start, node.end, node.comment))
        elif is_leaf(node) and node.trailing.strip().startswith("--"):
            num_spaces = node.trailing.index("--")
            text = node.trailing[num_spaces:]
            start = node.lean_file.offset(node.end, num_spaces)
            end = node.lean_file.offset(start, len(text))
            comments.append(Comment(start, end, text))

    ast.traverse_preorder(_callback, node_cls=None)
    return comments

def get_code_without_comments(
    lean_file: LeanFile, start: Pos, end: Pos, comments: List[Comment]
) -> str:
    base = start 
    code_segs = []  

    for c in comments:
        if base <= c.start and c.end <= end: 
            code_segs.append(lean_file[base : c.start])
            base = c.end 

    code_segs.append(lean_file[base:end])  
    code = "".join(code_segs) 

    code = _COMMENT_REGEX.sub("", code) 
    assert "--" not in code and "/-" not in code  

    return code.strip() 

@dataclass(frozen=True)
class TracedTheorem:
    """A traced theorem is a theorem with additional information such as the AST."""

    theorem: LeanTheorem 
    """The corresponding :class:`Theorem` object.
    """

    traced_file: 'TracedFile'

    ast: Union[CommandTheoremNode, LemmaNode, MathlibTacticLemmaNode] = field(
        repr=False, compare=False
    ) 
    """AST of the theorem.
    """

    comments: List[Comment] = field(repr=False, compare=False)
    """All comments in the theorem/proof.
    """

    @property
    def name(self) -> str:
        return self.theorem.name

    @property
    def start(self) -> Pos:
        return self.ast.start

    @property
    def end(self) -> Pos:
        return self.ast.end

    def get_proof_node(self) -> Node:
        return self.ast.get_proof_node()

    def locate_proof(self) -> Tuple[Pos, Pos]:
        start, end = self.get_proof_node().get_closure()
        if end < self.end:
            end = self.end
        return start, end
    
    @property
    def root_dir(self) -> Path:
        return self.theorem.root_dir
    
    @property
    def path(self) -> Path:
        return self.theorem.path
    
    @property
    def abs_path(self) -> Path:
        return self.theorem.abs_path

@dataclass(eq=False) 
class TracedFile:
    lean_file: LeanFile  
    """Lean source file of this traced file.
    """

    ast: FileNode = field(repr=False)  
    """Abstract syntax tree (AST) of the entire :code:`*.lean` file.
    """

    comments: List[Comment] = field(repr=False)  
    """All comments in the :code:`*.lean` file.
    """

    traced_theorems: Dict[str, List[TracedTheorem]] = field(default_factory=dict, repr=False) 

    def __post_init__(self) -> None:
        def _callback(
            node: Union[CommandTheoremNode, LemmaNode, MathlibTacticLemmaNode], _
        ) -> bool:
            if not isinstance(
                node,
                (
                    CommandTheoremNode,
                    LemmaNode,
                    MathlibTacticLemmaNode,
                ),
            ):
                return False

            theorem = LeanTheorem(self.lean_file, node.name)
            comments = self._filter_comments(node.start, node.end)
            self.traced_theorems.setdefault(node.name, []).append(
                TracedTheorem(theorem, self, node, comments)
            )
            return True

        self.ast.traverse_preorder(_callback, node_cls=None)

    @property
    def root_dir(self) -> Path:
        return self.lean_file.root_dir

    @property
    def path(self) -> Path:
        """Path of the :file:`*.lean` file relative to the root directory."""
        return self.lean_file.path

    @property
    def abs_path(self) -> Path:
        """Absolute path of the :code:`*.lean` file."""
        return self.lean_file.abs_path

    @classmethod
    def from_traced_file(
        cls, repo: LeanRepo, lean_file_path: Path, ast_json_path: Path
    ) -> "TracedFile":
        assert ast_json_path.exists()
        assert ast_json_path.suffixes == [
            ".ast",
            ".json",
        ], f"{ast_json_path} is not a *.ast.json file"
        lean_file = LeanFile(repo, lean_file_path)
        data = json.load(ast_json_path.open())
        ast = FileNode.from_data(data, lean_file)
        comments = _collect_lean4_comments(ast)
        return cls(lean_file, ast, comments)

    def get_traced_theorem(self, name: str, idx: int = 0) -> TracedTheorem:
        theorems = self.traced_theorems.get(name)
        if theorems is None:
            raise DojoInitError(f"No traced theorems found for name '{name}'.")
        if not (0 <= idx < len(theorems)):
            raise DojoInitError(
                f"Index {idx} out of range for traced theorems with name '{name}'."
            )
        return theorems[idx]
    
    def get_traced_theorems(self) -> Dict[str, List[TracedTheorem]]:
        return self.traced_theorems.copy()

    def _filter_comments(self, start: Pos, end: Pos) -> List[Comment]:
        comments = []  
        for c in self.comments: 
            if c.start < start:
                assert c.end <= start 
            elif c.start < end:  
                assert c.end <= end  
                comments.append(c) 
        return comments 

    @property
    def has_prelude(self) -> bool: 
        """Check whether the file starts with :code:``prelude``.

        :code:``prelude`` instructs Lean NOT to include its built-in library automatically.
        """
        result = False  

        def _callback(node: ModulePreludeNode, _: List[Node]):  
            nonlocal result
            result = True 
            return True 

        self.ast.traverse_preorder(_callback, node_cls=ModulePreludeNode)  
        return result  

@dataclass(eq=False) 
class TracedRepo:
    repo: LeanRepo
    traced_files: Dict[Path, TracedFile] = field(default_factory=dict, repr=False) 
    TRACE_FILE_NUM_THREADS: int = 1

    def __init__(
        self,
        repo_path: str,
        replica_repo_path: str = None,
        TRACE_FILE_NUM_THREADS: int = 1
    ):
        try:
            if isinstance(repo_path, str) and os.path.exists(repo_path) and os.path.isdir(repo_path):
                repo_path = os.path.abspath(repo_path)
                if replica_repo_path:
                    if isinstance(replica_repo_path, str):
                        replica_repo_path = os.path.abspath(replica_repo_path)
                        try:
                            shutil.copytree(repo_path, replica_repo_path)
                        except Exception as e:
                            raise RepoInitError(f"Failed to copy repository from '{repo_path}' to '{replica_repo_path}': {e}")
                        self.repo = LeanRepo(replica_repo_path)
                    else:
                        raise RepoInitError(f"Invalid replica_repo_path '{replica_repo_path}'.")
                else:
                    self.repo = LeanRepo(repo_path)
            else:
                raise RepoInitError(f"Invalid repo_path '{repo_path}'.")
            
            if isinstance(TRACE_FILE_NUM_THREADS, int) and TRACE_FILE_NUM_THREADS > 0:
                self.TRACE_FILE_NUM_THREADS = TRACE_FILE_NUM_THREADS
            else:
                raise RepoInitError(f"Invalid TRACE_FILE_NUM_THREADS value: {TRACE_FILE_NUM_THREADS}")

            self.traced_files = {}
            
            with working_directory(self.root_dir):
                logger.info("Begin excuting 'lake build'.")
                execute("lake build", RepoInitError)
                logger.info("Command `lake build` executed successfully.")
                if not os.path.exists(".initialized"):
                    result = execute("lean --print-prefix", RepoInitError, capture_output=True)
                    if not result or not result[0]:
                        raise RepoInitError(f"Failed to retrieve lean prefix.")
                    lean_prefix = result[0].strip()
                    dest = LEAN4_PACKAGES_DIR / "lean4"
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(lean_prefix, str(dest))
                    logger.info(f"Successfully copied Lean prefix from '{lean_prefix}' to '{dest}'.")
                    if os.path.exists(LEAN4_REPL_PATH.name):
                        os.remove(LEAN4_REPL_PATH.name)
                    shutil.copyfile(LEAN4_REPL_PATH, LEAN4_REPL_PATH.name)
                    if os.path.exists("lakefile.lean"):
                        with open("lakefile.lean", "r+", encoding="utf-8") as f:
                            content = f.read()
                            if "lean_lib Lean4Repl" not in content:
                                f.write("\nlean_lib Lean4Repl {\n\n}\n")
                                logger.info("Successfully appended Lean4Repl to lakefile.lean.")
                    else:
                        if not os.path.exists("lakefile.toml"):
                            raise RepoInitError("No lakefile.lean or lakefile.toml found.")
                        with open("lakefile.toml", "r+", encoding="utf-8") as f:
                            content = f.read()
                            if 'name = "Lean4Repl"' not in content:
                                f.write('\n[[lean_lib]]\nname = "Lean4Repl"\n')
                                logger.info("Successfully appended Lean4Repl to lakefile.toml.")
                    execute("lake build Lean4Repl", RepoInitError("Failed to build Lean4Repl. You may run into issues when interacting with the repo."))
                    with open(".initialized", "w") as f:
                        f.write("The repo initialization has been completed. If you want to reinitialize, please delete this file.")
                    logger.info("Repository initialized. To reinitialize, delete the '.initialized' marker file.")
                else:
                    logger.info("Repository already initialized. Skipping initialization steps.")
        except Exception as e:
            logger.error(e)
            raise

    def _get_traced_file(self, lean_file_path: Union[str, Path], use_cache = True) -> TracedFile:
        with working_directory(self.root_dir):
            if not isinstance(lean_file_path, str) and not isinstance(lean_file_path, Path):
                raise DojoInitError(f"Invalid lean_file_path '{lean_file_path}'.")
            lean_file_path = Path(lean_file_path)
            if lean_file_path.is_absolute():
                try:
                    lean_file_path = lean_file_path.relative_to(self.root_dir)
                except Exception as e:
                    raise DojoInitError(f"Invalid lean_file_path '{lean_file_path}': {e}")
            if not os.path.exists(lean_file_path) or not os.path.isfile(lean_file_path):
                raise DojoInitError(f"Invalid lean_file_path '{lean_file_path}'.")
            if lean_file_path.suffix != ".lean":
                raise DojoInitError(f"Invalid file extension for lean_file_path '{lean_file_path}': expected '.lean', but got '{lean_file_path.suffix}'.")
            ast_json_path = LEAN4_BUILD_DIR / "ir" / lean_file_path.with_suffix(".ast.json")
            if use_cache:
                if lean_file_path in self.traced_files:
                    return self.traced_files[lean_file_path]  
                if not os.path.exists(ast_json_path) or not os.path.isfile(ast_json_path):
                    cmd = f"lake env lean --threads {self.TRACE_FILE_NUM_THREADS} --run {LEAN4_DATA_EXTRACTOR_PATH} " + str(lean_file_path)
                    execute(cmd, DojoInitError(f"Failed to extract AST infos from '{lean_file_path}'. Please ensure that the input lean file is not a dependency (not in '{LEAN4_PACKAGES_DIR}')"))
            else:
                if lean_file_path in self.traced_files:
                    del self.traced_files[lean_file_path]
                if os.path.exists(ast_json_path) and os.path.isfile(ast_json_path):
                    os.remove(ast_json_path)
                cmd = f"lake env lean --threads {self.TRACE_FILE_NUM_THREADS} --run {LEAN4_DATA_EXTRACTOR_PATH} " + str(lean_file_path)
                execute(cmd, DojoInitError(f"Failed to extract AST infos from '{lean_file_path}'. Please ensure that the input lean file is not a dependency (not in '{LEAN4_PACKAGES_DIR}')"))
            
            self.traced_files[lean_file_path] = TracedFile.from_traced_file(self.repo, lean_file_path, ast_json_path)
            return self.traced_files[lean_file_path]

    def get_traced_theorem_from_file(self, lean_file_path: Union[str, Path], theorem_name: str, idx: int = 0, use_cache = True) -> TracedTheorem:
        try:
            traced_file = self._get_traced_file(lean_file_path, use_cache)
            return traced_file.get_traced_theorem(theorem_name, idx)
        except Exception as e:
            logger.error(e)
            raise
    
    def get_traced_theorems_from_file(self, lean_file_path: Union[str, Path], use_cache = True) -> Dict[str, List[TracedTheorem]]:
        try:
            traced_file = self._get_traced_file(lean_file_path, use_cache)
            return traced_file.get_traced_theorems()
        except Exception as e:
            logger.error(e)
            raise

    @property
    def root_dir(self) -> Path:
        return Path(self.repo.url)