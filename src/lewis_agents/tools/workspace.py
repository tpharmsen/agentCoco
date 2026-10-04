import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path


class WorkspaceTool:
    def __init__(
        self,
        root: Path,
        timeout_seconds: int,
        approve_command: Callable[[str, Path], bool] | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.approve_command = approve_command

    def _safe_path(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("Path is outside the configured coding workspace")
        return candidate

    def read_file(self, relative_path: str) -> str:
        return self._safe_path(relative_path).read_text()

    def write_file(self, relative_path: str, content: str) -> str:
        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Wrote {relative_path}"

    def run_command(self, command: str) -> str:
        if not command.strip():
            raise ValueError("Command must not be empty")
        if self.approve_command is None:
            raise PermissionError("Command execution requires explicit human approval")
        if not self.approve_command(command, self.root):
            return "Command was rejected by the human approver"
        completed = subprocess.run(
            shlex.split(command),
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode:
            raise RuntimeError(f"Command failed with exit code {completed.returncode}: {output}")
        return output
