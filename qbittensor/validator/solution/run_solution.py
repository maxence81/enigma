# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
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

import base64
import binascii
import os
import shutil
import subprocess
import time
import zipfile

try:
    import bittensor as bt
except ImportError:
    import logging as _logging
    class _BtDummy:
        logging = _logging.getLogger("qbittensor")
    bt = _BtDummy()

from qbittensor.validator.solution.exceptions.invalid_solution import InvalidSolutionError
from qbittensor.utils.solution_status import OutputExtractionStatus, ValidationFailureReason

from .docker_ops import DockerOps
from .solution_context import SolutionExecution
from qbittensor.challenges.solution_output import (
    RESULT_JSON_FILENAME,
    SOLUTION_LOG_FILENAME,
)
from .constants import (
    CHALLENGE_INPUT_DIRNAME,
    CONTAINER_CHALLENGE_INPUT_PATH,
    CONTAINER_OUTPUT_DIRNAME,
    CONTAINER_SOLUTION_DIRNAME,
    SOLUTION_OUTPUT_SEPARATOR,
    SOLUTION_OUTPUT_ZIP_FILENAME,
    SOLUTION_STDOUT_MAX_BYTES_DEFAULT,
    SOLUTION_STDOUT_MAX_BYTES_ENV,
    VALIDATOR_DOCKER_CAP_DROP_DEFAULT,
    VALIDATOR_DOCKER_CAP_DROP_ENV,
    VALIDATOR_DOCKER_CPU_LIMIT_DEFAULT,
    VALIDATOR_DOCKER_CPU_LIMIT_ENV,
    VALIDATOR_DOCKER_GPUS_DEFAULT,
    VALIDATOR_DOCKER_GPUS_ENV,
    VALIDATOR_DOCKER_MINER_USER_DEFAULT,
    VALIDATOR_DOCKER_MINER_USER_ENV,
    VALIDATOR_DOCKER_NO_NEW_PRIVILEGES_DEFAULT,
    VALIDATOR_DOCKER_NO_NEW_PRIVILEGES_ENV,
    VALIDATOR_DOCKER_PIDS_LIMIT_DEFAULT,
    VALIDATOR_DOCKER_PIDS_LIMIT_ENV,
    VALIDATOR_DOCKER_READ_ONLY_DEFAULT,
    VALIDATOR_DOCKER_READ_ONLY_ENV,
    VALIDATOR_DOCKER_TMPFS_DEFAULT,
    VALIDATOR_DOCKER_TMPFS_ENV,
    VALIDATOR_DOCKER_ULIMIT_NOFILE_DEFAULT,
    VALIDATOR_DOCKER_ULIMIT_NOFILE_ENV,
    VALIDATOR_MEMORY_LIMIT_DEFAULT,
    VALIDATOR_MEMORY_LIMIT_ENV,
    VALIDATOR_ZIP_MAX_UNCOMPRESSED_BYTES_DEFAULT,
    VALIDATOR_ZIP_MAX_UNCOMPRESSED_BYTES_ENV,
    VALIDATOR_CONTAINER_STOP_TIMEOUT_DEFAULT,
    VALIDATOR_CONTAINER_STOP_TIMEOUT_ENV,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _stdout_max_bytes() -> int:
    raw = os.getenv(SOLUTION_STDOUT_MAX_BYTES_ENV, str(SOLUTION_STDOUT_MAX_BYTES_DEFAULT)).strip()
    try:
        return int(raw)
    except ValueError:
        bt.logging.warning(
            f"Invalid {SOLUTION_STDOUT_MAX_BYTES_ENV}={raw!r}; "
            f"using default {SOLUTION_STDOUT_MAX_BYTES_DEFAULT} bytes."
        )
        return SOLUTION_STDOUT_MAX_BYTES_DEFAULT


def _solution_output_max_uncompressed_bytes() -> int:
    """Max allowed decompressed size for the miner's emitted solution artifacts zip.

    Uses the same env var as input solution zips for consistency.
    """
    raw = os.getenv(
        VALIDATOR_ZIP_MAX_UNCOMPRESSED_BYTES_ENV,
        str(VALIDATOR_ZIP_MAX_UNCOMPRESSED_BYTES_DEFAULT),
    ).strip()
    try:
        return int(raw)
    except ValueError:
        bt.logging.warning(
            f"Invalid {VALIDATOR_ZIP_MAX_UNCOMPRESSED_BYTES_ENV}={raw!r}; "
            f"using default {VALIDATOR_ZIP_MAX_UNCOMPRESSED_BYTES_DEFAULT} bytes."
        )
        return VALIDATOR_ZIP_MAX_UNCOMPRESSED_BYTES_DEFAULT


def prepare_challenge_input_mount_dir(workspace: str) -> str:
    """
    Create a fresh host directory for this run's challenge input (mounted ``:ro``).
    Challenge setup should write files only under this path.
    """
    mount_dir = os.path.join(os.path.abspath(workspace), CHALLENGE_INPUT_DIRNAME)
    if os.path.isdir(mount_dir):
        bt.logging.info(f"🗑️ Removing previous challenge input mount dir: {mount_dir}")
        try:
            shutil.rmtree(mount_dir)
            bt.logging.info(f"🗑️ Removed previous challenge input mount dir: {mount_dir}")
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to remove previous mount dir {mount_dir}: {e}")
    os.makedirs(mount_dir, mode=0o755)
    return mount_dir


def extract_stdout_output(container_ref: str, host_workspace: str) -> OutputExtractionStatus:
    """
    Pull a stopped miner container's output via ``docker logs`` and split it into
    the run's log file and the run's solution-output zip.

    The miner contract is:

        <text logs...>
        <SOLUTION_OUTPUT_SEPARATOR>
        <base64-encoded zip of the solution_artifacts directory>

    We capture the combined output from `docker logs` (CLI always mixes stdout+stderr).
    The full combined stream is written to the diagnostics log file.
    Protocol extraction uses a tolerant base64 decoder to handle possible trailing stderr junk.

    - Full raw docker logs (stdout + stderr) → ``<host_workspace>/output/container.log`` (SOLUTION_LOG_FILENAME)
    - Base64 payload → ``solution_artifacts.zip`` then extracted

    Returns an :class:`OutputExtractionStatus` describing whether artifacts were
    extracted or why they were not.
    """
    host_output = os.path.join(os.path.abspath(host_workspace), CONTAINER_OUTPUT_DIRNAME)
    os.makedirs(host_output, exist_ok=True)

    artifacts_dir = os.path.join(host_output, CONTAINER_SOLUTION_DIRNAME)
    os.makedirs(artifacts_dir, exist_ok=True)

    max_bytes = _stdout_max_bytes()
    full_logs_str = ""
    try:
        ops = DockerOps()

        # Plain docker logs (combined stdout+stderr). Full for diagnostics.
        # Protocol extraction tolerates trailing junk.
        full_logs_str = ops.logs(container_ref, check=True) or ""
        raw_stdout = full_logs_str.encode("utf-8", errors="replace") if isinstance(full_logs_str, str) else (full_logs_str or b"")
    except subprocess.CalledProcessError as e:
        stderr = ((e.stderr or e.stdout or b"").decode("utf-8", errors="replace")).strip()
        bt.logging.error(
            f"❌ Failed to read stdout from '{container_ref}': {stderr}"
        )
        _write_extraction_diagnostics(artifacts_dir, container_ref, f"docker logs failed: {stderr}")
        return OutputExtractionStatus.DOCKER_LOGS_FAILED
    except Exception as e:
        bt.logging.error(f"❌ Failed to invoke docker logs for '{container_ref}': {e}")
        _write_extraction_diagnostics(artifacts_dir, container_ref, f"Failed to invoke docker logs: {e}")
        return OutputExtractionStatus.DOCKER_LOGS_FAILED

    if full_logs_str:
        log_path = os.path.join(host_output, SOLUTION_LOG_FILENAME)
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(full_logs_str)
        except Exception:
            pass

    raw_stdout = raw_stdout or b""
    if len(raw_stdout) > max_bytes:
        bt.logging.error(
            f"❌ Container '{container_ref}' produced {len(raw_stdout)} bytes of stdout, "
            f"exceeding the {max_bytes} byte cap; truncating and refusing to extract artifacts."
        )
        _write_extraction_diagnostics(
            artifacts_dir, container_ref,
            f"stdout exceeded cap ({len(raw_stdout)} > {max_bytes} bytes)."
        )
        return OutputExtractionStatus.STDOUT_TOO_LARGE

    _, payload_b64, separator_found = _split_on_separator(raw_stdout)

    if not separator_found:
        bt.logging.warning(
            f"⚠️ No solution-output separator found in stdout of '{container_ref}'; "
            f"treating the entire stdout as logs and skipping artifact extraction."
        )
        _write_extraction_diagnostics(
            artifacts_dir, container_ref,
            "No SOLUTION_OUTPUT_SEPARATOR found in container stdout.\n"
            "The solution must write logs, then exactly this line on its own line:\n"
            f"{SOLUTION_OUTPUT_SEPARATOR.decode('utf-8', errors='replace')!r}\n"
            f"then base64-encoded zip of {RESULT_JSON_FILENAME} (and other files).\n"
            "See the mock_solution.py example for the required contract."
        )
        return OutputExtractionStatus.NO_SEPARATOR

    payload_b64 = payload_b64.strip()

    # Use shared tolerant cleaner (handles combined docker logs stream)
    from qbittensor.challenges.solution_output import clean_base64_payload
    payload_b64 = clean_base64_payload(payload_b64)

    try:
        payload_bytes = base64.b64decode(payload_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        bt.logging.error(
            f"❌ Solution payload from '{container_ref}' is not valid base64: {e}"
        )
        _write_extraction_diagnostics(artifacts_dir, container_ref, f"Base64 decode of solution payload failed: {e}")
        return OutputExtractionStatus.INVALID_BASE64

    zip_path = os.path.join(host_output, SOLUTION_OUTPUT_ZIP_FILENAME)
    try:
        with open(zip_path, "wb") as f:
            f.write(payload_bytes)
    except OSError as e:
        bt.logging.error(
            f"❌ Failed to write solution zip for '{container_ref}' to '{zip_path}': {e}"
        )
        _write_extraction_diagnostics(artifacts_dir, container_ref, f"Failed to write decoded solution zip: {e}")
        return OutputExtractionStatus.EXTRACT_FAILED

    if os.path.isdir(artifacts_dir):
        shutil.rmtree(artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)

    if not payload_bytes:
        bt.logging.warning(
            f"⚠️ Solution-output payload from '{container_ref}' is empty after the separator; "
            f"leaving '{artifacts_dir}' empty."
        )
        _write_extraction_diagnostics(
            artifacts_dir, container_ref,
            "Separator was present but the base64 payload after it was empty. "
            "The solution wrote the separator line but no (or zero-length) zip data after it."
        )
        return OutputExtractionStatus.EMPTY_PAYLOAD

    if not zipfile.is_zipfile(zip_path):
        bt.logging.error(
            f"❌ Solution-output payload from '{container_ref}' at '{zip_path}' "
            f"is not a valid zip archive; skipping artifact extraction."
        )
        _write_extraction_diagnostics(
            artifacts_dir, container_ref,
            f"Decoded payload after separator is not a valid zip file (path: {zip_path}). "
            "Common causes: logging after the separator, wrong base64, or the solution wrote raw binary instead of base64."
        )
        return OutputExtractionStatus.INVALID_ZIP

    max_uncompressed = _solution_output_max_uncompressed_bytes()
    try:
        declared = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    declared += info.file_size
        if declared > max_uncompressed:
            bt.logging.error(
                f"❌ Solution artifacts zip declares {declared} uncompressed bytes "
                f"(limit {max_uncompressed}); refusing extraction."
            )
            _write_extraction_diagnostics(
                artifacts_dir, container_ref,
                f"Declared uncompressed size {declared} exceeds limit {max_uncompressed}."
            )
            return OutputExtractionStatus.EXTRACT_FAILED
    except Exception as e:
        bt.logging.warning(f"Could not pre-estimate zip size, will rely on streaming limit: {e}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, artifacts_dir, max_bytes=max_uncompressed)
    except zipfile.BadZipFile as e:
        bt.logging.error(
            f"❌ Failed to read solution zip for '{container_ref}' at '{zip_path}': {e}"
        )
        _write_extraction_diagnostics(artifacts_dir, container_ref, f"BadZipFile when extracting: {e}")
        return OutputExtractionStatus.INVALID_ZIP
    except (OSError, RuntimeError) as e:
        bt.logging.error(
            f"❌ Failed to extract solution zip for '{container_ref}' to '{artifacts_dir}': {e}"
        )
        _write_extraction_diagnostics(artifacts_dir, container_ref, f"Failed to extract solution zip: {e}")
        return OutputExtractionStatus.EXTRACT_FAILED

    bt.logging.info(
        f"✅ Extracted solution output from '{container_ref}' to '{artifacts_dir}'"
    )
    return OutputExtractionStatus.ARTIFACTS_EXTRACTED


def resolve_verif_json(artifacts_dir: str) -> str | None:
    """Return the canonical <workspace>/verif.json path for this solution.

    Challenge setup writes verif.json to the workspace root (sibling to
    challenge_input_mount and output/). We know the exact layout, so we
    compute it directly instead of walking upward from the miner-controlled
    artifacts directory.

    artifacts_dir is expected to be <workspace>/output/solution_artifacts.
    """
    try:
        p = os.path.abspath(artifacts_dir)
        output_dir = os.path.dirname(p)
        ws_root = os.path.dirname(output_dir)
        candidate = os.path.join(ws_root, "verif.json")
        if os.path.isfile(candidate):
            return candidate
        return None
    except Exception:
        return None


def _split_on_separator(raw_stdout: bytes) -> tuple[bytes, bytes, bool]:
    """
    Split a stdout byte stream around the first occurrence of
    :data:`SOLUTION_OUTPUT_SEPARATOR`.

    Returns ``(logs_bytes, payload_bytes, separator_found)``. The separator itself
    is consumed (it appears in neither the logs nor the payload).
    """
    idx = raw_stdout.find(SOLUTION_OUTPUT_SEPARATOR)
    if idx == -1:
        return raw_stdout, b"", False
    return raw_stdout[:idx], raw_stdout[idx + len(SOLUTION_OUTPUT_SEPARATOR):], True


def _write_extraction_diagnostics(artifacts_dir: str, container_ref: str, message: str) -> None:
    """Write a small diagnostics file inside the solution artifacts directory.

    This ensures that even on extraction failures, useful diagnostic information
    (e.g. missing separator, bad base64, docker logs failure, etc.) is included
    when the artifacts directory is zipped and uploaded on the failure path.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    diag_path = os.path.join(artifacts_dir, "extraction_diagnostics.txt")
    try:
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(f"Container: {container_ref}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
            f.write(message)
            f.write("\n")
    except Exception as e:
        bt.logging.warning(f"⚠️ Failed to write extraction diagnostics file: {e}")


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    """Return True if the ZipInfo indicates a symlink (best-effort across platforms)."""
    # On Unix, high 16 bits of external_attr are the st_mode; 0o120000 is S_IFLNK.
    mode = (info.external_attr >> 16) & 0o170000
    if mode == 0o120000:
        return True
    return False


def _safe_extract_zip(
    zf: zipfile.ZipFile, destination: str, *, max_bytes: int | None = None
) -> None:
    """
    Extract a zip while enforcing:
      - No absolute paths or ``..`` traversal (lexical).
      - No symlinks (reject ZipInfo symlink entries + post-extraction sweep).
      - Optional running byte budget on *actual* decompressed bytes written.

    Raises RuntimeError on any policy violation. The destination may contain
    partial files on error; callers should clean up the workspace.
    """
    destination_abs = os.path.abspath(destination)
    os.makedirs(destination_abs, exist_ok=True)

    total_written = 0
    extracted_paths: list[str] = []

    for member in zf.infolist():
        member_path = os.path.normpath(os.path.join(destination_abs, member.filename))
        if not (
            member_path == destination_abs
            or member_path.startswith(destination_abs + os.sep)
        ):
            raise RuntimeError(
                f"Refusing to extract '{member.filename}': escapes destination directory"
            )

        if _is_zip_symlink(member):
            raise RuntimeError(
                f"Refusing to extract '{member.filename}': symlinks are not allowed in archives"
            )

        if member.is_dir():
            os.makedirs(member_path, exist_ok=True)
            continue

        parent = os.path.dirname(member_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with zf.open(member, "r") as src, open(member_path, "wb") as dst:
            while True:
                chunk = src.read(64 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                total_written += len(chunk)
                if max_bytes is not None and total_written > max_bytes:
                    raise RuntimeError(
                        f"Zip extraction exceeded size limit ({total_written} > {max_bytes} bytes)"
                    )

        extracted_paths.append(member_path)

    for root, dirs, files in os.walk(destination_abs):
        for name in dirs + files:
            full = os.path.join(root, name)
            if os.path.islink(full):
                try:
                    os.unlink(full)
                except Exception:
                    pass
                raise RuntimeError(
                    f"Refusing archive: symlink detected after extraction at '{os.path.relpath(full, destination_abs)}'"
                )


def _container_run_user() -> str | None:
    """Return the ``docker run --user`` value, or None if non-root enforcement is disabled."""
    raw = os.getenv(
        VALIDATOR_DOCKER_MINER_USER_ENV,
        VALIDATOR_DOCKER_MINER_USER_DEFAULT,
    )
    if raw is None:
        return VALIDATOR_DOCKER_MINER_USER_DEFAULT or None
    user = raw.strip()
    return user or None


def docker_run_security_args() -> list[str]:
    """
    Extra ``docker run`` flags that harden miner solution containers.
    Each setting is controlled by an environment variable (see constants.py).
    """
    args: list[str] = []

    stop_timeout = os.getenv(VALIDATOR_CONTAINER_STOP_TIMEOUT_ENV, VALIDATOR_CONTAINER_STOP_TIMEOUT_DEFAULT).strip()
    if stop_timeout:
        args.extend(["--stop-timeout", stop_timeout])

    pids_limit = os.getenv(VALIDATOR_DOCKER_PIDS_LIMIT_ENV, VALIDATOR_DOCKER_PIDS_LIMIT_DEFAULT).strip()
    if pids_limit:
        args.extend(["--pids-limit", pids_limit])

    ulimit_nofile = os.getenv(
        VALIDATOR_DOCKER_ULIMIT_NOFILE_ENV, VALIDATOR_DOCKER_ULIMIT_NOFILE_DEFAULT
    ).strip()
    if ulimit_nofile:
        args.extend(["--ulimit", f"nofile={ulimit_nofile}"])

    cap_drop = os.getenv(VALIDATOR_DOCKER_CAP_DROP_ENV, VALIDATOR_DOCKER_CAP_DROP_DEFAULT).strip()
    if cap_drop:
        args.extend(["--cap-drop", cap_drop])

    if _env_bool(VALIDATOR_DOCKER_NO_NEW_PRIVILEGES_ENV, VALIDATOR_DOCKER_NO_NEW_PRIVILEGES_DEFAULT):
        args.extend(["--security-opt", "no-new-privileges:true"])

    if _env_bool(VALIDATOR_DOCKER_READ_ONLY_ENV, VALIDATOR_DOCKER_READ_ONLY_DEFAULT):
        args.append("--read-only")

    tmpfs = os.getenv(VALIDATOR_DOCKER_TMPFS_ENV, VALIDATOR_DOCKER_TMPFS_DEFAULT).strip()
    if tmpfs:
        args.extend(["--tmpfs", tmpfs])

    cpus = os.getenv(VALIDATOR_DOCKER_CPU_LIMIT_ENV, VALIDATOR_DOCKER_CPU_LIMIT_DEFAULT).strip()
    if cpus:
        args.extend(["--cpus", cpus])

    memory = os.getenv(VALIDATOR_MEMORY_LIMIT_ENV, VALIDATOR_MEMORY_LIMIT_DEFAULT).strip()
    if memory:
        args.extend(["--memory", memory, "--memory-swap", memory])

    gpus = os.getenv(VALIDATOR_DOCKER_GPUS_ENV, VALIDATOR_DOCKER_GPUS_DEFAULT).strip()
    if gpus:
        args.extend(["--gpus", gpus])

    run_user = _container_run_user()
    if run_user:
        args.extend(["--user", run_user])

    return args


def run_image_detached(
    image_name: str,
    container_name: str,
    validator_label: str,
    challenge_input_mount_dir: str,
    *,
    solution_execution: SolutionExecution | None = None,
) -> str:
    """
    Run a Docker container in detached mode with hardened mounts.

    - Challenge input: fresh host dir per run, read-only at ``/challenge_input``.
    - Solution output: nothing is mounted into the container. The miner writes
      text logs + a :data:`SOLUTION_OUTPUT_SEPARATOR` line + a base64-encoded
      zip of artifacts to stdout. The validator captures this after the container
      exits using ``docker logs`` (see :func:`extract_stdout_output`).

    If ``solution_execution`` is provided, stable identifiers (including the
    solution_id) are attached as ``--label`` values via ``solution_execution.to_labels()``.
    This is the preferred path. Direct calls from tests that don't need our
    correlation labels can pass ``solution_execution=None``.
    """
    challenge_mount = os.path.abspath(challenge_input_mount_dir)

    cmd = [
        "docker",
        "run",
        "-d",
        "--network",
        "none",
        *docker_run_security_args(),
        "--name",
        container_name,
        "-v", f"{challenge_mount}:{CONTAINER_CHALLENGE_INPUT_PATH}:ro",
        "--label",
        validator_label,
    ]

    if solution_execution is not None:
        for k, v in solution_execution.to_labels().items():
            cmd.extend(["--label", f"{k}={v}"])

    cmd.append(image_name)

    try:
        ops = DockerOps(validator_label=validator_label)
        container_id = ops.run(
            cmd[1:],  # strip the leading "docker" because DockerOps.run() prepends it
            description=f"docker run for container {container_name}",
        )

        bt.logging.info(
            f"🚀 Container '{container_name}' started in background (ID: {container_id}) "
            f"with label '{validator_label}'"
        )
        return container_id

    except InvalidSolutionError:
        # Already has rich diagnostics from DockerOps
        raise

    except Exception as e:
        bt.logging.error(f"❌ An unexpected error occurred while starting the container: {e}")
        raise InvalidSolutionError(
            message=f"Unexpected error starting Docker container: {e}. "
                    "Check validator logs for the full docker command and any additional context.",
            failure_reason=ValidationFailureReason.RUN_FAILURE,
        ) from e
