"""Full-weight IPC sync for SGLang via the ``checkpoint_engine`` protocol.

Colocate-only: the trainer and SGLang engine share the same node. Uses
``checkpoint_engine.worker.update_weights_from_ipc`` (ZMQ REQ/REP + CUDA IPC
shared buffer) for zero-copy weight transfer — the receiver reconstructs
tensor **views** into the shared buffer and calls ``model.load_weights()``
in-place, so no extra GPU memory is allocated on the rollout side.

This is the SGLang analogue of :class:`~unirl.distributed.weight_sync.full.ipc.IPCWeightSync`
(which is vLLM-Omni only). The protocols differ fundamentally — see
:class:`~unirl.distributed.weight_sync.transfer.ckpt_engine_transfer.CkptEngineWeightSender`.

Thread discipline:
    - **NativeBackend**: ``engine.update_weights_from_ipc()`` calls
      ``loop.run_until_complete()`` which MUST run on the engine's thread.
      So the receiver runs in the **main thread** and the sender in a
      **daemon thread** (inverted from IPCWeightSync).
    - **HTTPBackend**: ``update_from_ipc`` is a blocking HTTP POST (thread-safe).
      The receiver runs in a **daemon thread** and the sender in the
      **main thread** (same as IPCWeightSync).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync

logger = logging.getLogger(__name__)


class CkptEngineIPCWeightSync(FullWeightSync):
    """Colocate full-weight sync for SGLang via checkpoint_engine IPC.

    Uses SGLang's ``update_weights_from_ipc`` API (ZMQ + CUDA IPC shared buffer)
    for zero-copy weight transfer. The trainer allocates a double buffer on its
    GPU, shares it via CUDA IPC, and sends bucket metadata over ZMQ. Each SGLang
    scheduler subprocess (one per TP rank) creates a REP socket, reconstructs
    tensor views into the shared buffer, and calls ``load_weights`` (which
    handles TP sharding internally).
    """

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        bucket_size_mb: int = 2048,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        super().__init__(
            backend=backend,
            bucket_size_mb=bucket_size_mb,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            adapter_name=adapter_name,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        self._rollout = rollout

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Push full weights to the SGLang engine via checkpoint_engine IPC.

        Runs on every train rank (BROADCAST); the ``_iter_full_tensors`` walk
        all-gathers each FSDP shard in lockstep. Only ``tp_rank==0`` (the rank
        hosting the SGLang engine) actually pushes; others do the all-gather
        collective but skip the IPC push.
        """
        ri = self.rank_info
        rank = ri.rank if ri is not None else 0
        is_tp_zero = ri is None or ri.tp_rank == 0

        if not is_tp_zero:
            # Still run the all-gather collective (lockstep), but skip the push.
            for _ in self._iter_full_tensors():
                pass
            logger.debug(
                "[CkptEngine-IPC] rank %s: all-gathered weights but skipped push (tp_rank=%s/%s)",
                rank,
                ri.tp_rank if ri else 0,
                ri.tp_size if ri else 1,
            )
            return

        # Enumerate all TP rank GPU UUIDs and construct ZMQ socket paths.
        tp_size = self._get_tp_size()
        base_gpu_id = self._get_base_gpu_id()
        zmq_handles = self._build_zmq_handles(tp_size, base_gpu_id)

        logger.info(
            "[CkptEngine-IPC] rank %s: pushing full weights to %d TP rank(s) via checkpoint_engine IPC",
            rank,
            tp_size,
        )

        recv_error: dict = {}

        def _spawn_receiver() -> None:
            """Trigger the SGLang engine to connect its REP sockets."""
            try:
                self._rollout.update_weights_from_ipc(
                    zmq_handles=zmq_handles,
                    flush_cache=self._flush_cache,
                )
            except Exception as exc:
                recv_error["exc"] = exc

        # Determine backend type for thread assignment.
        # NativeBackend: engine.loop.run_until_complete must run on engine's
        #   thread → receiver in main thread, sender in daemon thread.
        # HTTPBackend: HTTP POST is thread-safe → receiver in daemon thread,
        #   sender in main thread (same as existing IPCWeightSync).
        is_native = self._is_native_backend()

        if is_native:
            # NativeBackend: receiver in main thread, sender in daemon thread.
            sender_thread = threading.Thread(target=self._run_sender, args=(zmq_handles,), daemon=True)
            sender_thread.start()
            try:
                _spawn_receiver()  # blocks until all TP workers finish loading
            finally:
                sender_thread.join()
        else:
            # HTTPBackend: receiver in daemon thread, sender in main thread.
            recv_thread = threading.Thread(target=_spawn_receiver, daemon=True)
            recv_thread.start()
            try:
                self._run_sender(zmq_handles)
            finally:
                recv_thread.join()

        if "exc" in recv_error:
            raise RuntimeError("CkptEngineIPCWeightSync: rollout receiver failed") from recv_error["exc"]

        self.weight_version += 1
        logger.info("[CkptEngine-IPC] rank %s: full weight sync completed", rank)

    def _run_sender(self, zmq_handles: Dict[str, str]) -> None:
        """Run the CkptEngineWeightSender in a daemon thread."""
        from unirl.distributed.weight_sync.transfer.ckpt_engine_transfer import (
            CkptEngineWeightSender,
        )

        sender = CkptEngineWeightSender(
            zmq_handles=zmq_handles,
            bucket_size_mb=self._bucket_bytes // (1024 * 1024),
        )
        asyncio.run(sender.async_send_weights(self._iter_full_tensors()))

    def _get_tp_size(self) -> int:
        """Get the SGLang engine's TP size."""
        tp_size = getattr(self._rollout, "_tp_size", 1)
        return int(tp_size) if tp_size else 1

    def _get_base_gpu_id(self) -> int:
        """Get the first GPU ID of the SGLang engine's TP group."""
        device_ids = getattr(self._rollout, "_tp_device_ids", None)
        if device_ids:
            return int(device_ids[0])
        return 0

    def _build_zmq_handles(self, tp_size: int, base_gpu_id: int) -> Dict[str, str]:
        """Build the ``{device_uuid: zmq_socket_path}`` dict for all TP ranks.

        SGLang scheduler subprocesses use ``torch.cuda.current_device()`` (a
        0-based local index within their own CUDA_VISIBLE_DEVICES) to look up
        their UUID in ``zmq_handles``. The trainer process may not see all GPUs
        (Ray assigns 1 GPU per worker), so we read GPU UUIDs from the NVIDIA
        sysfs instead of ``torch.cuda.get_device_properties``.

        Each SGLang scheduler subprocess sees exactly 1 GPU (its own), so
        ``torch.cuda.current_device()`` returns 0 there, and
        ``get_device_properties(0).uuid`` returns that GPU's UUID. The trainer
        must provide a mapping that covers all TP rank GPU UUIDs.
        """
        import subprocess

        job_id = os.environ.get("RAY_JOB_ID", "default")
        handles: Dict[str, str] = {}

        # Read all GPU UUIDs from nvidia-smi (works regardless of CUDA_VISIBLE_DEVICES)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            all_uuids = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        except Exception:
            # Fallback: use torch if available (may fail if not all GPUs visible)
            all_uuids = []
            try:
                import torch as _torch

                for i in range(_torch.cuda.device_count()):
                    all_uuids.append(str(_torch.cuda.get_device_properties(i).uuid))
            except Exception:
                pass

        # Use the first tp_size UUIDs (SGLang uses base_gpu_id..base_gpu_id+tp_size-1)
        for i in range(tp_size):
            gpu_idx = base_gpu_id + i
            if gpu_idx < len(all_uuids):
                # nvidia-smi returns UUID with "GPU-" prefix (e.g. "GPU-74334149-..."),
                # which matches SGLang's get_device_uuid(): f"GPU-{torch.uuid}" where
                # torch.uuid is "GPU-74334149-..." (also has "GPU-" prefix).
                uuid_str = all_uuids[gpu_idx]
            else:
                # Fallback: use index-based key (won't match SGLang's UUID lookup,
                # but allows the test to proceed)
                uuid_str = f"GPU-idx-{gpu_idx}"
            socket_path = f"ipc:///tmp/unirl-ckpt-engine-{job_id}-gpu-{gpu_idx}.sock"
            handles[uuid_str] = socket_path

        logger.info(
            "[CkptEngine-IPC] Built %d ZMQ handles for TP ranks (base_gpu_id=%d): %s",
            len(handles),
            base_gpu_id,
            list(handles.keys())[:3],  # show first 3 for debugging
        )
        return handles

    def _is_native_backend(self) -> bool:
        """Check if the rollout uses NativeBackend (vs HTTPBackend)."""
        backend = getattr(self._rollout, "_backend", None)
        if backend is None:
            return False
        return type(backend).__name__ == "NativeBackend"


__all__ = ["CkptEngineIPCWeightSync"]
