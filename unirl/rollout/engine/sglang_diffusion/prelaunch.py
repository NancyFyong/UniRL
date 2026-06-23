"""Driver-side pre-launch of the SGLang diffusion server.

Why this exists
---------------
``sglang.multimodal_gen.runtime.launch_server.launch_server`` blocks on a bare
``mp.Pipe`` ``reader.recv()`` (launch_server.py:178, no timeout). When that
function runs *inside a Ray Worker actor*, the pipe's write-end fd is inherited
by sibling Ray processes (forked actor model), so ``recv()`` sees an open writer
and blocks forever — the scheduler child HAS sent ``{"status":"ready"}`` but the
parent never sees it. See the plan
``/root/.codebuddy/plans/blazing-vortex-lovelace.md`` Bug A for the full
root-cause trace.

The fix is architectural: launch SGLang as a ``subprocess.Popen`` from the
**driver process** (before ``DevicePool.create_remote`` forks the Ray actor),
then have each Worker actor connect to it in ``local_mode=False`` (ZMQ client,
no mp.Pipe). A clean ``subprocess.Popen`` has no sibling Ray processes, so the
fd-inheritance bug does not apply — ``launch_server``'s ``mp.Pipe`` works as
designed (the parent closes its write-ends at launch_server.py:163-164, leaving
the children as the only writers).

Public surface
--------------
- :func:`launch_sglang_server_process` — driver-side: spawn + ping-wait.
- :func:`terminate` — graceful SIGTERM → 10s grace → SIGKILL.
- ``python -m unirl.rollout.engine.sglang_diffusion.prelaunch <intent.pkl>``
  — the subprocess entry point (the shim that builds ServerArgs and calls
  ``launch_server``).

The shim installs the UniRL hijack when ``SGLANG_DIFFUSION_PATCHES=1`` is set
in the env, so the pre-launched server has the RL weight-sync handlers.
"""

from __future__ import annotations

import logging
import os
import pickle
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default host for the pre-launched server. The Ray Worker actors connect to it
# over ZMQ; they run on the same node (the anchor device's node), so loopback is
# sufficient. Override via the intent's ``host`` field for cross-node setups.
_DEFAULT_HOST = "127.0.0.1"

# How long to wait for the scheduler to bind its port before giving up.
# Overridable via env var ``UNIRL_SGLANG_PRELAUNCH_TIMEOUT_S`` (seconds, float).
# Default 600s: SD3.5-medium (~5GB) on 8× H200 with CUDA graph compilation can
# exceed the previous 180s default, especially on a cold cache.
import os as _os
_PING_TIMEOUT_S = float(_os.environ.get("UNIRL_SGLANG_PRELAUNCH_TIMEOUT_S", "600.0"))
_PING_INTERVAL_S = 1.0


def _find_free_port() -> int:
    """Bind to 0, read the assigned port, close. Race-free per-call."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_DEFAULT_HOST, 0))
        return s.getsockname()[1]


def _port_is_open(host: str, port: int) -> bool:
    """True if a TCP connect to (host, port) succeeds (zmq REP has bound)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def launch_sglang_server_process(
    server_intent: Dict[str, Any],
    *,
    device_ids: List[int],
    python_executable: Optional[str] = None,
) -> Tuple[subprocess.Popen, str, int]:
    """Spawn the SGLang diffusion server as a subprocess from the driver.

    Args:
        server_intent: the dict from
            :meth:`SGLangDiffusionEngineConfig.server_intent` (model path,
            parallelism, ports, adapter extras). ``host`` / ``scheduler_port``
            are filled in here if unset.
        device_ids: physical GPU ids the server should pin to (set as
            ``CUDA_VISIBLE_DEVICES`` for the subprocess).
        python_executable: Python binary to use (defaults to ``sys.executable``).

    Returns:
        ``(popen, host, scheduler_port)`` — the driver holds the ``Popen`` and
        calls :func:`terminate` on shutdown; each Worker actor connects to
        ``host:scheduler_port`` in ``local_mode=False``.

    Raises:
        RuntimeError: if the scheduler port does not accept connections within
            :data:`_PING_TIMEOUT_S` seconds, or if the subprocess exits first.
    """
    # Resolve host + scheduler_port (bind-to-0 if the config didn't pin them).
    host = str(server_intent.get("host") or _DEFAULT_HOST)
    scheduler_port = server_intent.get("scheduler_port")
    if scheduler_port is None:
        scheduler_port = _find_free_port()
    # Patch the intent so the shim builds ServerArgs with the resolved values.
    server_intent = dict(server_intent)
    server_intent["host"] = host
    server_intent["scheduler_port"] = int(scheduler_port)

    # Pickle the intent to a temp file the subprocess reads back. Avoids
    # thousands of CLI args and the quoting/typing hazards of a big ServerArgs.
    fd, intent_path = tempfile.mkstemp(prefix="sglang_prelaunch_", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(server_intent, f)
        return _spawn_and_wait(intent_path, host, int(scheduler_port), device_ids, python_executable)
    finally:
        try:
            os.unlink(intent_path)
        except OSError:
            pass


def _spawn_and_wait(
    intent_path: str,
    host: str,
    scheduler_port: int,
    device_ids: List[int],
    python_executable: Optional[str],
) -> Tuple[subprocess.Popen, str, int]:
    py = python_executable or sys.executable
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
    # Ask the shim to install the UniRL hijack (RL weight-sync handlers). The
    # hijack's wrap_mp_process_for_children re-installs patches in the
    # scheduler's spawn children — correct and desired.
    env["SGLANG_DIFFUSION_PATCHES"] = "1"

    cmd = [py, "-m", "unirl.rollout.engine.sglang_diffusion.prelaunch", intent_path]
    log_prefix = f"[sglang-prelaunch host={host} port={scheduler_port} gpus={device_ids}]"
    logger.info("%s launching: %s", log_prefix, " ".join(cmd))

    # Redirect subprocess stdout/stderr to a log file so we can diagnose
    # crashes / slow startup even after the timeout kills the process.
    # (PIPE would be lost on terminate; the file survives.)
    prelaunch_log_path = _os.environ.get(
        "UNIRL_SGLANG_PRELAUNCH_LOG",
        f"/tmp/sglang_prelaunch_{host}_{scheduler_port}.log",
    )
    prelaunch_log = open(prelaunch_log_path, "wb")
    logger.info("%s stdout/stderr -> %s", log_prefix, prelaunch_log_path)

    popen = subprocess.Popen(
        cmd,
        env=env,
        stdout=prelaunch_log,
        stderr=subprocess.STDOUT,
        # New session so SIGTERM reaches the whole process tree (scheduler
        # spawns its own children).
        start_new_session=True,
    )

    deadline = time.monotonic() + _PING_TIMEOUT_S
    while time.monotonic() < deadline:
        if popen.poll() is not None:
            # Process exited early — drain the log file for the error and raise.
            prelaunch_log.flush()
            try:
                with open(prelaunch_log_path, "rb") as f:
                    out = f.read()
            except OSError:
                out = b""
            raise RuntimeError(
                f"{log_prefix} subprocess exited with code {popen.returncode} "
                f"before the scheduler bound its port. Log file: "
                f"{prelaunch_log_path}\nOutput:\n"
                f"{out.decode('utf-8', errors='replace')}"
            )
        if _port_is_open(host, scheduler_port):
            logger.info("%s scheduler is up.", log_prefix)
            return popen, host, scheduler_port
        time.sleep(_PING_INTERVAL_S)

    # Timed out — kill the subprocess and report.
    terminate(popen)
    raise RuntimeError(
        f"{log_prefix} scheduler did not bind {host}:{scheduler_port} within "
        f"{_PING_TIMEOUT_S}s. Log file: {prelaunch_log_path}"
    )


def terminate(popen: subprocess.Popen, *, grace_s: float = 10.0) -> None:
    """SIGTERM the process group, wait ``grace_s``, SIGKILL if still alive."""
    if popen.poll() is not None:
        return
    try:
        # start_new_session=True put the child in its own session; kill the
        # whole group so the scheduler's spawn children die too.
        os.killpg(popen.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        popen.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(popen.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            popen.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("sglang prelaunch process %d did not die after SIGKILL.", popen.pid)


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def _main(intent_path: str) -> None:
    """The shim: build ServerArgs from the intent, call launch_server, block.

    Runs in a clean ``subprocess.Popen`` (not a Ray actor), so the
    ``launch_server`` mp.Pipe fd-inheritance bug does not apply — the parent's
    write-ends are closed at launch_server.py:163-164, leaving the spawn
    children as the only writers, and ``reader.recv()`` returns promptly.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(intent_path, "rb") as f:
        intent: Dict[str, Any] = pickle.load(f)

    # Install the UniRL hijack (RL weight-sync handlers) when the driver asked
    # for it. The shim runs in a clean process, so the hijack's
    # wrap_mp_process_for_children patches mp.Process globally BEFORE any
    # scheduler spawn — the patches propagate into the spawn children correctly.
    if os.environ.get("SGLANG_DIFFUSION_PATCHES") == "1":
        logger.info("Installing UniRL SglangDiffusionHijack in prelaunch subprocess.")
        from unirl.rollout.engine.sglang_diffusion._patches.hijack import (
            SglangDiffusionHijack,
        )

        SglangDiffusionHijack.hijack()

    import dataclasses

    from sglang.multimodal_gen.runtime.launch_server import launch_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    allowed = {f.name for f in dataclasses.fields(ServerArgs)}
    server_kwargs = {k: v for k, v in intent.items() if k in allowed}
    server_args = ServerArgs.from_kwargs(**server_kwargs)

    # launch_http_server=False — the Worker actors talk ZMQ to the scheduler,
    # not HTTP. launch_server spawns the scheduler workers, waits for "ready"
    # via mp.Pipe (works here: no Ray sibling fd inheritance), and returns the
    # process list. The shim stays alive to keep the daemon children alive.
    logger.info(
        "prelaunch shim calling launch_server(model_path=%s, host=%s, scheduler_port=%s)",
        getattr(server_args, "model_path", None),
        getattr(server_args, "host", None),
        getattr(server_args, "scheduler_port", None),
    )
    processes = launch_server(server_args, launch_http_server=False)
    logger.info("launch_server returned %d worker process(es).", len(processes))

    # Block until any worker exits. The daemon children keep running as long as
    # this process lives; when it exits, they die with it (daemon=True).
    import multiprocessing as mp

    mp.connection.wait([p.sentinel for p in processes])
    # If we get here, a worker died. Report and exit non-zero so the driver
    # notices (the ping loop above will also see the port close).
    for i, p in enumerate(processes):
        if p.exitcode is not None and p.exitcode != 0:
            logger.error("prelaunch worker %d exited with code %s.", i, p.exitcode)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: python -m {__name__} <intent.pkl>", file=sys.stderr)
        sys.exit(2)
    _main(sys.argv[1])
