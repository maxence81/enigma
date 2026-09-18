# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including but not limited to
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
DockerOps - centralized, testable helper for all Docker CLI interactions in the validator.

This extracts the repeated raw `subprocess.run(["docker", ...])` calls that were
scattered across solution_container_manager.py, run_solution.py, build_docker_image.py,
and validate_docker_image.py.

Benefits:
- Consistent capture (text), logging, timeouts, and rich error messages.
- Easier to mock in unit tests (one place instead of patching subprocess.run everywhere).
- Label-aware filtering helpers that are safe for multi-validator hosts.
- Clear separation between "generic docker command" and "validator solution container operations".
"""

import subprocess
from typing import Optional

try:
    import bittensor as bt
except ImportError:
    import logging as _logging
    class _BtDummy:
        logging = _logging.getLogger("qbittensor")
    bt = _BtDummy()

from .exceptions.invalid_solution import InvalidSolutionError
from qbittensor.utils.solution_status import ValidationFailureReason


class DockerOps:
    """Helper for running Docker commands with consistent behavior and good diagnostics."""

    def __init__(self, validator_label: Optional[str] = None):
        self.validator_label = validator_label

    # ------------------------------------------------------------------
    # Core runner (replaces the old _run_docker_command)
    # ------------------------------------------------------------------
    def _run(
        self,
        cmd: list[str],
        *,
        description: str = "docker command",
        check: bool = False,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a docker CLI command.

        - Always captures stdout/stderr as text.
        - On failure (non-zero exit or 'docker' not found when check=True), raises
          InvalidSolutionError with a rich, platform-friendly message containing the
          full command, exit code, stderr, and stdout.
        - Logs the command at debug level.
        """
        try:
            bt.logging.debug(f"🐳 Running {description}: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                timeout=timeout,
            )
            return result

        except FileNotFoundError as e:
            msg = (
                f"Docker CLI not found while running {description}. "
                "The 'docker' executable is not available in the PATH of the validator process. "
                "Is Docker installed and properly integrated (especially on WSL)?"
            )
            bt.logging.error(f"❌ {msg} | command={' '.join(cmd)} | {e}")
            raise InvalidSolutionError(
                message=msg,
                failure_reason=ValidationFailureReason.RUN_FAILURE,
            ) from e

        except subprocess.CalledProcessError as e:
            returncode = e.returncode
            stderr = (e.stderr or "").strip() or "(no stderr captured)"
            stdout = (e.stdout or "").strip() or "(no stdout captured)"

            if returncode == 127:
                detail = f"Docker command not found (exit status 127) while running {description}"
            else:
                detail = f"{description} failed with exit code {returncode}"

            bt.logging.error(
                f"❌ {detail}\n"
                f"   Command: {' '.join(cmd)}\n"
                f"   stderr:\n{stderr}\n"
                f"   stdout:\n{stdout}"
            )

            platform_msg = (
                f"{detail}\n\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {returncode}\n\n"
                f"stderr:\n{stderr}\n\n"
                f"stdout:\n{stdout}"
            )
            raise InvalidSolutionError(
                message=platform_msg,
                failure_reason=ValidationFailureReason.RUN_FAILURE,
            ) from e

        except subprocess.TimeoutExpired as e:
            bt.logging.error(f"❌ Timeout while running {description}: {' '.join(cmd)}")
            if check:
                raise InvalidSolutionError(
                    message=f"Timeout running {description}",
                    failure_reason=ValidationFailureReason.WALL_TIME_FAILURE,
                ) from e
            raise

    # ------------------------------------------------------------------
    # High-level convenience methods used heavily by SolutionContainerManager
    # ------------------------------------------------------------------
    def ps(
        self,
        *,
        all: bool = False,
        status: Optional[str] = None,
        format: str = "{{.ID}}",
    ) -> list[str]:
        """List containers (optionally filtered by our validator label)."""
        cmd = ["docker", "ps"]
        if all:
            cmd.append("-a")
        if self.validator_label:
            cmd.extend(["--filter", f"label={self.validator_label}"])
        if status:
            cmd.extend(["--filter", f"status={status}"])
        cmd.extend(["--format", format])

        res = self._run(cmd, description="docker ps", check=True)
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]

    def stop(self, name_or_id: str) -> bool:
        res = self._run(["docker", "stop", name_or_id], description=f"stop {name_or_id}", check=False)
        if res.returncode == 0:
            bt.logging.info(f"🛑 Stopped container {name_or_id}")
            return True
        bt.logging.warning(f"⚠️ Stop may have failed for {name_or_id}: {(res.stderr or res.stdout or '').strip()}")
        return False

    def rm(self, name_or_id: str, volumes: bool = False) -> bool:
        cmd = ["docker", "rm", "-v" if volumes else "", name_or_id]
        cmd = [c for c in cmd if c]  # drop empty
        res = self._run(cmd, description=f"rm {name_or_id}", check=False)
        if res.returncode == 0:
            bt.logging.info(f"🗑️ Removed container {name_or_id}")
            return True
        bt.logging.warning(f"⚠️ Failed to remove container {name_or_id}: {(res.stderr or res.stdout or '').strip()}")
        return False

    def rmi(self, image_ref: str, force: bool = False) -> bool:
        cmd = ["docker", "rmi", "-f" if force else "", image_ref]
        cmd = [c for c in cmd if c]
        res = self._run(cmd, description=f"rmi {image_ref}", check=False)
        if res.returncode == 0:
            bt.logging.info(f"🗑️ Removed image {image_ref}")
            return True
        bt.logging.warning(f"⚠️ Failed to remove image {image_ref}: {(res.stderr or res.stdout or '').strip()}")
        return False

    def run(self, args: list[str], description: str = "docker run") -> str:
        """
        Convenience for launch-style commands: runs and returns stripped stdout.
        Raises rich InvalidSolutionError on failure.
        """
        full_cmd = ["docker"] + args
        result = self._run(full_cmd, description=description, check=True)
        return result.stdout.strip()

    def run_command(
        self,
        args: list[str],
        *,
        description: str = "docker command",
        check: bool = False,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """
        Public low-level runner for arbitrary docker subcommands.
        Example:
            ops.run_command(["image", "inspect", "foo", "--format", "{{.Size}}"])
        """
        full_cmd = ["docker"] + args
        return self._run(full_cmd, description=description, check=check, timeout=timeout)

    def inspect(self, identifier: str, fmt: str) -> Optional[str]:
        """docker inspect <identifier> --format <fmt>"""
        res = self.run_command(
            ["inspect", identifier, "--format", fmt],
            description=f"docker inspect {identifier}",
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip() or None
        return None

    def image_inspect(self, image_name: str, fmt: str) -> Optional[str]:
        """docker image inspect <image_name> --format <fmt>"""
        res = self.run_command(
            ["image", "inspect", image_name, "--format", fmt],
            description=f"docker image inspect {image_name}",
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip() or None
        return None

    def logs(self, container_ref: str, tail: Optional[int] = None, check: bool = False, stdout_only: bool = False) -> str:
        """docker logs <container_ref>

        Note: docker logs CLI always returns the combined stdout+stderr stream.
        Callers that need a clean protocol payload should use clean_base64_payload().
        """
        cmd = ["logs", container_ref]
        if tail is not None:
            cmd.extend(["--tail", str(tail)])
        res = self.run_command(cmd, description=f"docker logs {container_ref}", check=check)
        # Always combined
        return (res.stdout or "") + (res.stderr or "")

    def build(self, context_dir: str, tags: list[str] | None = None, **extra_args) -> subprocess.CompletedProcess:
        """
        docker build -t <tag> ... <context_dir>
        Returns the CompletedProcess (caller can inspect returncode, stdout, etc.).
        For streaming builds with custom log files, consider using run_command directly
        or enhancing this method.
        """
        cmd = ["build"]
        for tag in (tags or []):
            cmd.extend(["-t", tag])
        cmd.append(context_dir)
        return self.run_command(cmd, description=f"docker build {context_dir}", check=False)

    def build_image(self, context_dir: str, tags: list[str] | None = None, **extra_args) -> subprocess.CompletedProcess:
        """Convenience wrapper for building images (alias to build for clarity)."""
        return self.build(context_dir, tags=tags, **extra_args)

    def run_container(
        self,
        image_name: str,
        *,
        gpus: str | None = None,
        rm: bool = True,
        **extra_args,
    ) -> subprocess.CompletedProcess:
        """
        Run a container with common options.
        Example: ops.run_container("myimg", gpus="all", rm=True)
        Extra args can be passed for other flags.
        """
        cmd = ["run"]
        if rm:
            cmd.append("--rm")
        if gpus:
            cmd.extend(["--gpus", gpus])
        cmd.append(image_name)
        return self.run_command(cmd, description=f"docker run {image_name}", check=False)

    def popen(self, args: list[str], **popen_kwargs):
        """Low-level Popen for advanced streaming cases (e.g. live build logs)."""
        full = ["docker"] + args
        return subprocess.Popen(full, **popen_kwargs)

    # ------------------------------------------------------------------
    # Prune helpers (used by the resource prune timer)
    # ------------------------------------------------------------------
    def prune_builder(self, until: str) -> subprocess.CompletedProcess:
        cmd = ["docker", "builder", "prune", "-f", "--filter", f"until={until}"]
        return self._run(cmd, description="docker builder prune", check=False, timeout=300)

    def prune_images(self, until: str) -> subprocess.CompletedProcess:
        cmd = ["docker", "image", "prune", "-a", "-f", "--filter", f"until={until}"]
        return self._run(cmd, description="docker image prune", check=False, timeout=300)

    # ------------------------------------------------------------------
    # Version / availability
    # ------------------------------------------------------------------
    @classmethod
    def check_available(cls) -> bool:
        """Delegate to the shared startup check (CLI + BuildKit/buildx)."""
        # Imported lazily to avoid a circular import at module load time.
        from qbittensor.validator.solution.solution_container_manager import is_docker_available

        return is_docker_available()
