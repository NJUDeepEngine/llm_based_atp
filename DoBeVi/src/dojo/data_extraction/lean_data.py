import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Generator
from ..errors import RepoInitError
from ..constants import _LEAN4_VERSION_REGEX

@dataclass(eq=True, unsafe_hash=True)
class Pos:
    line_nb: int
    """Line number"""
    column_nb: int
    """Column number"""

    @classmethod
    def from_str(cls, s: str) -> "Pos":
        """Construct a :class:`Pos` object from its string representation, e.g., :code:`"(323, 1109)"`."""
        assert s.startswith("(") and s.endswith(")"), f"Invalid string representation of a position: {s}"
        line, column = s[1:-1].split(",")
        line_nb = int(line)
        column_nb = int(column)
        return cls(line_nb, column_nb)

    def __iter__(self) -> Generator[int, None, None]:
        yield self.line_nb
        yield self.column_nb

    def __repr__(self) -> str:
        return repr(tuple(self))

    def __lt__(self, other):
        return self.line_nb < other.line_nb or (
            self.line_nb == other.line_nb and self.column_nb < other.column_nb
        )

    def __le__(self, other):
        return self < other or self == other

def is_lean4_version_supported(v) -> bool:
    """Check if ``v`` is at least `v4.3.0-rc2`."""
    if not v.startswith("v"):
        return False
    v = v[1:]
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if major < 4 or (major == 4 and minor < 3):
        return False
    if (
        major > 4
        or (major == 4 and minor > 3)
        or (major == 4 and minor == 3 and patch > 0)
    ):
        return True
    assert major == 4 and minor == 3 and patch == 0
    if "4.3.0-rc" in v:
        rc = int(v.split("-")[1][2:])
        return rc >= 2
    else:
        return True

@dataclass(frozen=True)
class LeanRepo:
    url: str
    lean_version: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if os.path.basename(self.url) == "lean4":
            raise RepoInitError("Repo's name can not be 'lean4'.")
        lean_version = self._get_lean4_version()
        if not is_lean4_version_supported(lean_version):
            raise RepoInitError(f"Unsupported Lean version: '{lean_version}'")
        else:
            object.__setattr__(self, "lean_version", lean_version)

    def _get_lean4_version(self) -> str:
        toolchain_path = os.path.join(self.url, "lean-toolchain")
        if not os.path.isfile(toolchain_path):
            raise RepoInitError(f"'lean-toolchain' file not found in: '{self.url}'")
        try:
            with open(toolchain_path, "r") as f:
                toolchain = f.read()
        except Exception as e:
            raise RepoInitError(f"Failed to read 'lean-toolchain' file: {e}")

        m = _LEAN4_VERSION_REGEX.fullmatch(toolchain.strip())
        if not m:
            raise RepoInitError(f"Invalid lean-toolchain config: '{toolchain.strip()}'")

        lean4_version = m["version"]
        if not lean4_version.startswith("v") and lean4_version[0].isnumeric():
            lean4_version = "v" + lean4_version
        return lean4_version

@dataclass(frozen=True)
class LeanFile:
    repo: LeanRepo

    path: Path

    code: List[str] = field(init=False, repr=False)
    """Raw source code as a list of lines."""

    endwith_newline: bool = field(init=False, repr=False)
    """Whether the last line ends with a newline."""

    num_bytes: List[int] = field(init=False, repr=False)
    """The number of UTF-8 bytes of each line, including newlines."""

    def __post_init__(self) -> None:
        assert self.root_dir.is_absolute(), f"Root directory must be an absolute path: {self.root_dir}"
        assert self.path.suffix == ".lean", f"File extension must be .lean: {self.path}"
        assert not self.path.is_absolute(), f"Path must be a relative path: {self.path}"

        code = []
        endwith_newline = None
        num_bytes = []

        for line in self.abs_path.open("rb"):
            if b"\r\n" in line:
                raise RuntimeError(
                    f"{self.abs_path} contains Windows-style line endings. This is discouraged (see https://github.com/leanprover-community/mathlib4/pull/6506)."
                )
            if line.endswith(b"\n"):
                endwith_newline = True
                line = line[:-1]
            else:
                endwith_newline = False
            code.append(line.decode("utf-8"))
            num_bytes.append(len(line) + 1)

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "endwith_newline", endwith_newline)
        object.__setattr__(self, "num_bytes", num_bytes)

    @property
    def root_dir(self) -> Path:
        return Path(self.repo.url)

    @property
    def abs_path(self) -> Path:
        """Absolute path of a :class:`LeanFile` object.
        """
        return self.root_dir / self.path

    @property
    def num_lines(self) -> int:
        """Number of lines in a source file."""
        return len(self.code)

    def num_columns(self, line_nb: int) -> int:
        """Number of columns in a source file."""
        return len(self.get_line(line_nb))

    @property
    def start_pos(self) -> Pos:
        """Return the start position of a source file.

        Returns:
            Pos: A :class:`Pos` object representing the start of this file.
        """
        return Pos(1, 1)

    @property
    def end_pos(self) -> Pos:
        """Return the end position of a source file.

        Returns:
            Pos: A :class:`Pos` object representing the end of this file.
        """
        if self.is_empty():
            return self.start_pos
        line_nb = self.num_lines
        column_nb = 1 + len(self.code[-1])
        return Pos(line_nb, column_nb)

    def is_empty(self) -> bool:
        return len(self.code) == 0

    def convert_pos(self, byte_idx: int) -> Pos:
        """Convert a byte index (:code:`String.Pos` in Lean 4) to a :class:`Pos` object."""
        n = 0
        for i, num_bytes in enumerate(self.num_bytes, start=1):
            n += num_bytes
            if n == byte_idx and i == self.num_lines:
                byte_idx -= 1
            if n > byte_idx:
                line_byte_idx = byte_idx - (n - num_bytes)
                if line_byte_idx == 0:
                    return Pos(i, 1)

                line = self.get_line(i)
                m = 0

                for j, c in enumerate(line, start=1):
                    m += len(c.encode("utf-8"))
                    if m >= line_byte_idx:
                        return Pos(i, j + 1)

        raise ValueError(f"Invalid byte index {byte_idx} in {self.path}.")

    def offset(self, pos: Pos, delta: int) -> Pos:
        """Off set a position by a given number."""
        line_nb, column_nb = pos
        num_columns = len(self.get_line(line_nb)) - column_nb + 1
        if delta <= num_columns:
            return Pos(line_nb, column_nb + delta)
        delta_left = delta - num_columns - 1

        for i in range(line_nb, self.num_lines):
            line = self.code[i]
            l = len(line)
            if delta_left <= l:
                return Pos(i + 1, delta_left + 1)
            delta_left -= l + 1

        if delta_left == 0 and self.endwith_newline:
            return Pos(self.num_lines + 1, 1)

        raise ValueError(f"Invalid offset {delta} in {self.path}: {pos}.")

    def get_line(self, line_nb: int) -> str:
        """Return a given line of the source file.

        Args:
            line_nb (int): Line number (1-indexed).
        """
        return self.code[line_nb - 1]

    def __getitem__(self, key) -> str:
        """Return a code segment given its start/end positions.

        This enables ``lean_file[start:end]``.

        Args:
            key (slice): A slice of two :class:`Pos` objects for the start/end of the code segment.
        """
        assert isinstance(key, slice) and key.step is None
        if key.start is None:
            start_line = start_column = 1
        else:
            start_line, start_column = key.start
        if key.stop is None:
            end_line = self.num_lines
            end_column = 1 + len(self.get_line(end_line))
        else:
            end_line, end_column = key.stop
        if start_line == end_line:
            assert start_column <= end_column
            return self.get_line(start_line)[start_column - 1 : end_column - 1]
        else:
            assert start_line < end_line
            code_slice = [self.code[start_line - 1][start_column - 1 :]]
            for line_nb in range(start_line + 1, end_line):
                code_slice.append(self.get_line(line_nb))
            code_slice.append(self.get_line(end_line)[: end_column - 1])
            return "\n".join(code_slice)

@dataclass(frozen=True)
class LeanTheorem:
    file: LeanFile
    """Lean source file the theorem comes from."""

    name: str
    """Name of the theorem."""

    @property
    def root_dir(self) -> Path:
        return self.file.root_dir

    @property
    def path(self) -> Path:
        return self.file.path
    
    @property
    def abs_path(self) -> Path:
        return self.file.abs_path