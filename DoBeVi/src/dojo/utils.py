import os
import subprocess
from pathlib import Path
from contextlib import contextmanager
from typing import Type, Tuple, Union, List, Generator, Optional
from .constants import LEAN4_BUILD_DIR, _CAMEL_CASE_REGEX

@contextmanager
def working_directory(
    path: Union[Path, str] = None
) -> Generator[Path, None, None]:
    origin = Path.cwd()
    path = Path(path)
    assert path.exists()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(origin)

def execute(
    cmd: Union[str, List[str]],
    err: Optional[Union[Type[Exception], Exception]] = None,
    capture_output: bool = True,
) -> Optional[Tuple[str, str]]:
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            check=True,
        )
        if not capture_output:
            return None
        output = res.stdout.decode(errors="replace")
        error = res.stderr.decode(errors="replace")
        return output, error
    except Exception as e:
        if err is None:
            raise RuntimeError(f"Command failed: {cmd}\nCause: {e}") from e
        elif isinstance(err, Exception):
            raise err from e
        elif isinstance(err, type) and issubclass(err, Exception):
            raise err(f"Command `{cmd}` failed: {e}") from e
        else:
            raise TypeError(f"`err` must be Exception or Exception class, got: {type(err)}")

def camel_case(s: str) -> str:
    """Convert the string ``s`` to camel case."""
    return _CAMEL_CASE_REGEX.sub(" ", s).title().replace(" ", "")