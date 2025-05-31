import re
import os
import json
import psutil
import pexpect
import tempfile
from loguru import logger
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Dict, Any, Optional, TextIO
from ..utils import working_directory
from ..data_extraction.traced_data import TracedTheorem, get_code_without_comments
from ..errors import DojoInitError, DojoTacticTimeoutError, DojoCrashError
from ..constants import BLOCK_RE

@dataclass(frozen=True)
class TacticState:
    pp: str
    id: int = field(compare=False)
    message: Optional[str] = field(default=None, compare=False)

@dataclass(frozen=True)
class ProofFinished:
    id: int
    message: Optional[str] = field(default=None, compare=False)

@dataclass(frozen=True)
class ProofGivenUp:
    id: int

@dataclass(frozen=True)
class LeanError:
    id: int
    error: str

NormalResult = Union[
    TacticState,
    ProofFinished
]

TacticResult = Union[
    TacticState,
    ProofFinished,
    LeanError,
    ProofGivenUp,
]

def kill_descendants(pid: int) -> None:
    try:
        _kill_descendants(psutil.Process(pid))
    except psutil.NoSuchProcess:
        pass

def _kill_descendants(proc: psutil.Process) -> None:
    for child in proc.children():
        _kill_descendants(child)
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass

class Dojo:
    root_dir: Path = None
    entry: Union[Tuple[TracedTheorem, int], Tuple[str, str]] = None
    timeout: int = 30
    DOJO_NUM_THREADS: int = 1
    DOJO_MEMORY_LIMIT: int = 32

    states: Dict[int, NormalResult] = None
    interaction_path: List[Tuple[int, str]] = None
    invalid_state_cnt: int = 0

    modified_file: TextIO = None
    proc: pexpect.spawn = None
    
    def __init__(
        self, 
        root_dir: Path, 
        entry: Union[TracedTheorem, Tuple[str, str]], 
        timeout: int = 30,
        DOJO_NUM_THREADS: int = 1,
        DOJO_MEMORY_LIMIT: int = 32
    ) -> None:
        try:
            with working_directory(root_dir):
                self.root_dir = root_dir
                self.entry = entry

                if isinstance(timeout, int) and timeout > 0:
                    self.timeout = timeout
                else:
                    raise DojoInitError(f"Invalid timeout value '{timeout}'.")
                
                if isinstance(DOJO_NUM_THREADS, int) and DOJO_NUM_THREADS > 0:
                    self.DOJO_NUM_THREADS = DOJO_NUM_THREADS
                else:
                    raise DojoInitError(f"Invalid DOJO_NUM_THREADS value '{DOJO_NUM_THREADS}'")
                
                if isinstance(DOJO_MEMORY_LIMIT, int) and DOJO_MEMORY_LIMIT > 0:
                    self.DOJO_MEMORY_LIMIT = DOJO_MEMORY_LIMIT
                else:
                    raise DojoInitError(f"Invalid DOJO_MEMORY_LIMIT value '{DOJO_MEMORY_LIMIT}'")
                
                self.states = {}
                self.interaction_path = []
            
        except Exception as e:
            logger.error(e)
            raise

    def __enter__(self) -> "Dojo":
        try:
            if isinstance(self.entry, TracedTheorem):
                traced_theorem = self.entry
                self._set_modified_file_by_traced_theorem(traced_theorem)
            else:
                code, theorem_name = self.entry
                self._set_modified_file_by_code(code, theorem_name)
            self.spawn_pexpect()
            return self
        
        except Exception as e:
            logger.error(e)
            raise

    def _set_modified_file_by_traced_theorem(self, traced_theorem: TracedTheorem) -> None:
        os.makedirs(traced_theorem.abs_path.parent / "temp_files", exist_ok=True)
        self.modified_file = tempfile.NamedTemporaryFile(
            "wt",
            prefix=traced_theorem.path.stem,
            suffix=traced_theorem.path.suffix,
            dir=traced_theorem.abs_path.parent / "temp_files",
            delete=True,
        ).__enter__()     
        # logger.info(f"Modifying `{traced_theorem.path}` into `{self.modified_file.name}`")
        proof_start, proof_end = traced_theorem.locate_proof()
        lean_file = traced_theorem.theorem.file

        code_proof = "by\n  lean_dojo_repl\n  sorry\n"
        code_before_theorem = get_code_without_comments(
            lean_file, lean_file.start_pos, traced_theorem.start, traced_theorem.traced_file.comments
        )
        
        code_thereom = get_code_without_comments(
            lean_file, traced_theorem.start, proof_start, traced_theorem.traced_file.comments
        ).strip()
        
        if code_thereom.endswith(" where"):
            raise DojoInitError(
                "Cannot interact with theorems with the `where` keyword."
            )
        code_thereom += " " if code_thereom.endswith(":=") else " := "
        modified_code = (
            "import Lean4Repl\n"
            + code_before_theorem
            + "\n\nset_option maxHeartbeats 0 in\n"
            + code_thereom
            + code_proof
            + lean_file[proof_end:]
        )
        self.modified_file.write(modified_code)
        self.modified_file.flush()
    
    def _set_modified_file_by_code(self, code: str, theorem_name: Optional[str] = None) -> None:
        os.makedirs(self.root_dir / "temp_files")
        self.modified_file = tempfile.NamedTemporaryFile(
            "wt",
            prefix="tmp_thm",
            suffix=".lean",
            dir=self.root_dir / "temp_files",
            delete=True,
        ).__enter__()
        matches = list(BLOCK_RE.finditer(code))
        if not matches:
            raise DojoInitError(f"No theorem blocks matched the expected format.")
        if theorem_name is None:
            target_match = matches[-1]
        else:
            target_match = next(
                (m for m in matches if m.group('name') == theorem_name),
                None
            )
            if target_match is None:
                raise DojoInitError(f"No theorem named '{theorem_name}' was found.")
        out, last_end = [], 0
        for m in matches:
            if m != target_match:                          
                out.append(code[last_end:m.start()])
            else:
                before = code[last_end:m.start()]
                needs_opt = (
                    before.rstrip().splitlines()[-1:]
                    != ["set_option maxHeartbeats 0 in"]
                )
                block_text = m.group('block')
                indent = re.search(r"\n([ \t]*)\bsorry\b", block_text)
                indent = indent.group(1) if indent else "  "
                patched_block = re.sub(
                    r"([ \t]*)\bsorry\b",
                    rf"\n{indent}lean_dojo_repl\n{indent}sorry",
                    block_text,
                    count=1
                )
                if needs_opt:
                    patched_block = "set_option maxHeartbeats 0 in\n" + patched_block
                out.append(before + patched_block)
            last_end = m.end()

        out.append(code[last_end:])
        modified_code = "import Lean4Repl\n" + re.sub(r"\n{3,}", "\n\n", "".join(out)).strip() + "\n"
        self.modified_file.write(modified_code)
        self.modified_file.flush()

    def spawn_pexpect(self) -> None:
        with working_directory(self.root_dir):
            memory_limit = 1024 * self.DOJO_MEMORY_LIMIT
            modified_path = Path(self.modified_file.name).relative_to(self.root_dir)
            cmd = f"lake env lean --threads={self.DOJO_NUM_THREADS} --memory={memory_limit} {modified_path}"
            try:
                self.proc = pexpect.spawn(
                    cmd, timeout=self.timeout, encoding="utf-8", echo=False
                ) 
            except Exception as e:
                raise DojoInitError(f"Failed to spawn lean process: {e}")
        
        try:
            res = json.loads(self._read_next_line()[0]) 
        except Exception as ex:
            if isinstance(self.entry, TracedTheorem) and self.entry.traced_file.has_prelude:
                raise DojoInitError(
                    "Currently LeanDojo does not support interacting with proofs in prelude files."
                )
            elif isinstance(ex, EOFError):
                raise DojoInitError(f"Unexpected EOF during getting init_state: {ex}")
            elif isinstance(ex, pexpect.TIMEOUT):
                raise DojoInitError("The current timeout setting is insufficient to compile the Lean file and obtain the initial state. Please increase the timeout duration.")
            else:
                raise DojoInitError(ex)
        
        if res["error"]:
            raise DojoInitError("Error occurred during getting init_state.")
        if res["tacticState"] == "no goals":
            raise DojoInitError("The goal of the theorem can't be 'no goals'")
        self.states[0] = TacticState(
            self._post_process(res["tacticState"]),
            res["sid"],
        )

    def _check_alive(self) -> None:
        if self.proc.isalive():
            return
        exit_code = self.proc.exitstatus
        output = self.proc.before if hasattr(self.proc, "before") else ""
        raise EOFError(f"Process crashed (exit code: {exit_code}). Output before crash:\n{output}")

    def _read_next_line(self) -> Tuple[str, str]:
        """Read the next line from `self.proc`.

        Raises:
            EOFError: _description_
            DojoCrashError: _description_
            DojoInitError: _description_

        Returns:
            str: _description_
        """
        _REPL_PROMPT = "REPL>"
        msg: List[str] = []
        while True:
            try:
                index = self.proc.expect(["\n", f"{_REPL_PROMPT}.*?\n"])
                if index == 0:
                    if self.proc.before == "":
                        raise EOFError("Unexpected EOF: subprocess output was empty before newline.")
                    else:
                        msg.append(self.proc.before.strip())
                        continue
                self._check_alive()
                res = self.proc.match.string[len(_REPL_PROMPT) :].strip()
                return res, "\n".join(msg) + self.proc.before
            except pexpect.EOF as e:
                raise EOFError(f"Subprocess unexpectedly closed (pexpect.EOF) during REPL interaction.")
            except pexpect.TIMEOUT:
                raise

    def _post_process(self, tactic_state: str) -> str:
        """Post-process the pretty-printed tactic state.

        Args:
            tactic_state (str): _description_

        Returns:
            str: _description_
        """
        m = re.match(r"\d+ goals\n", tactic_state)
        if m is not None:
            return tactic_state[m.end() :]
        else:
            return tactic_state
    
    def _submit_request(self, req: str) -> Dict[str, Any]:
        """Submit a request to Lean and get the response.

        Args:
            req (str): _description_

        Raises:
            DojoCrashError: _description_

        Returns:
            Dict[str, Any]: _description_
        """
        self._check_alive()
        # logger.info(f"Submit request: {req}")
        self.proc.sendline(req)
        res, msg = self._read_next_line()
        result: Dict[str, Any] = json.loads(res)
        result["message"] = msg
        return result

    def run_tac(self, state: TacticState, tactic: str) -> TacticResult:
        try:
            return self._run_tac(state, tactic)
        except Exception as e:
            logger.error(e)
            if self.proc:
                kill_descendants(self.proc.pid)
                self.proc = None
            self._restore_iter_env()
            return LeanError(self.next_invalid_state_id, f"Lean process terminated abnormally: {e}")         
    
    def _run_tac(self, state: TacticState, tactic: str) -> TacticResult:
        try:
            if self.proc is None:
                logger.error(f"There is currently no running Lean process for theorem proving (possibly due to a previous crash or timeout). Please restart the process.")
                return None
            if not isinstance(state, TacticState):
                logger.warning(f"Attempting to run a tactic on an invalid state {state}.")
                return LeanError(self.next_invalid_state_id, f"Attempting to run a tactic on an invalid state {state}.")
            if not isinstance(tactic, str):
                logger.warning(f"Invalid tactic {tactic}")
                return LeanError(self.next_invalid_state_id, f"Invalid tactic {tactic}")
            tsid = state.id
            req = json.dumps({"sid": tsid, "cmd": tactic}, ensure_ascii=False)
            res = self._submit_request(req)
            if res["error"] is not None:
                if "proof contains `sorry`" in res["error"]:
                    return ProofGivenUp(self.next_invalid_state_id)
                else:
                    # logger.warning(f"Error occurred when running tactic '{tactic}'.")
                    return LeanError(self.next_invalid_state_id, res["error"].strip())
            elif res["tacticState"] == "no goals":
                assert res["sid"] > self.max_state_id
                self.interaction_path.append([tsid, tactic])
                self.states[res["sid"]] = ProofFinished(res["sid"], res["message"])
                return self.states[res["sid"]]
            else:
                if res["sid"] in self.states:
                    return self.states[res["sid"]]
                assert res["sid"] > self.max_state_id
                tactic_state = TacticState(
                    self._post_process(res["tacticState"]),
                    res["sid"],
                    res["message"],
                )
                self.interaction_path.append([tsid, tactic])
                self.states[res["sid"]] = tactic_state
                return tactic_state
            
        except Exception as ex:
            if isinstance(ex, EOFError):
                if self.proc:
                    kill_descendants(self.proc.pid)
                    self.proc = None
                raise DojoCrashError(f"The Lean process encountered an EOF error and has been terminated: {ex}")
            elif isinstance(ex, pexpect.TIMEOUT):
                if self.proc:
                    kill_descendants(self.proc.pid)
                    self.proc = None
                raise DojoTacticTimeoutError(f"The Lean process remained unresponsive after {self.timeout} seconds and has been terminated.")
            else:
                logger.error(ex)
                raise

    def _restore_iter_env(self) -> None:
        logger.info("Start restoring the interactive environment.")
        assert self.modified_file
        try:
            self.spawn_pexpect()
            for [sid, tactic] in self.interaction_path:
                req = json.dumps({"sid": sid, "cmd": tactic}, ensure_ascii=False)
                res = self._submit_request(req)
                assert res["error"] is None
            logger.info("Successfully restored interactive environment.")
        except Exception as e:
            raise DojoCrashError(f"Failed to restore interactive environment: {e}")
    
    @property
    def init_state(self) -> Optional[TacticState]:
        return self.states.get(0)
    
    @property
    def max_state_id(self) -> int:
        return len(self.states) - 1

    @property
    def next_invalid_state_id(self) -> int:
        self.invalid_state_cnt += 1
        return -self.invalid_state_cnt

    def print_states(self) -> None:
        for state_id, state in self.states.items():
            print(f"State ID: {state_id}")
            if isinstance(state, TacticState):
                print(f"Type    : TacticState")
                print(f"pp      :", end=" ")
                for i, line in enumerate(state.pp.splitlines()):
                    if i == 0:
                        print(f"{line}")
                    else:
                        print(f"          {line}")
                if state.message:
                    print(f"message : {state.message}")
            elif isinstance(state, ProofFinished):
                print(f"Type    : ProofFinished")
                if state.message:
                    print(f"message : {state.message}")
            print("-" * 40)

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        """Exit Dojo.

        Args:
            exc_type (None): _description_
            exc_val (None): _description_
            exc_tb (None): _description_
        """
        if self.proc:
            kill_descendants(self.proc.pid)
            self.proc = None
        if self.modified_file:
            self.modified_file.__exit__(exc_type, exc_val, exc_tb)
            self.modified_file = None
        self.entry = None
        self.interaction_path.clear()
        self.states.clear()