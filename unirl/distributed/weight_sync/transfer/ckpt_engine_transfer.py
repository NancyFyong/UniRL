"""Weight transfer via ZMQ + CUDA IPC using the ``checkpoint_engine`` protocol.

This sender is compatible with SGLang's ``update_weights_from_ipc`` path, which
delegates to ``checkpoint_engine.worker.update_weights_from_ipc``. The protocol
differs from verl's ``BucketedWeightSender`` in several ways (see plan):

- **Double buffer**: ``bucket_size * 2`` with absolute offsets
- **Metadata as ``list[dict]``**: NOT ``dict[str, dict]`` with ``is_last``
- **Termination**: two ``None`` signals (release + post_hook), not ``is_last``
- **No per-tensor IPC handle**: all tensors must fit in the bucket buffer
- **Multi-receiver fan-out**: one shared buffer, N REQ sockets (one per TP rank)

The sender creates N REQ sockets (one per TP rank GPU UUID), allocates ONE
double buffer on the trainer's GPU, and sends the same IPC handle + bucket
metadata to all N sockets. Each TP rank's scheduler subprocess creates a REP
socket, rebuilds the buffer view via IPC, and calls ``model.load_weights()``
(which does TP sharding internally).
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict, Iterator, List, Tuple

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

logger = logging.getLogger(__name__)


class CkptEngineWeightSender:
    """Send model weights via the checkpoint_engine ZMQ + CUDA IPC protocol.

    Creates one REQ socket per TP rank (all bound to distinct paths), allocates
    a shared double buffer, and fans the same IPC handle + bucket metadata to
    all sockets. Each receiver reconstructs a view into the shared buffer and
    calls ``load_weights`` (which handles TP sharding internally).

    Args:
        zmq_handles: Dict mapping device UUID to ZMQ socket path (one per TP rank).
        bucket_size_mb: Communication buffer size in MB (per half of double buffer).
    """

    def __init__(
        self,
        zmq_handles: Dict[str, str],
        bucket_size_mb: int = 2048,
    ) -> None:
        self.zmq_handles = dict(zmq_handles)
        self.socket_paths = list(self.zmq_handles.values())
        self.n_sockets = len(self.socket_paths)
        self.bucket_size_mb = int(bucket_size_mb)
        self.bucket_size = self.bucket_size_mb << 20

        self.zmq_context = zmq.Context.instance()
        self.sockets: List[zmq.Socket] = []
        self.buffer = None  # double buffer: bucket_size * 2

    async def async_send_weights(self, weights: Iterator[Tuple[str, "object"]]) -> None:
        """Send weights to all TP rank receivers.

        Args:
            weights: Generator yielding (name, tensor) pairs.
        """
        try:
            self._init_sockets()
            self._init_buffer()

            gidx = 0
            offset = 0
            bucket_meta: List[Dict[str, Any]] = []

            for name, weight in weights:
                weight_nbytes = weight.nbytes

                # Flush current bucket if this tensor doesn't fit
                if offset + weight_nbytes > self.bucket_size and bucket_meta:
                    torch.cuda.synchronize()
                    self._send_bucket(bucket_meta)
                    gidx += 1
                    offset = 0
                    bucket_meta = []

                assert offset + weight_nbytes <= self.bucket_size, (
                    f"Weight {name}({weight.shape}, {weight.dtype}) is too large "
                    f"to fit in the bucket ({weight_nbytes} > {self.bucket_size}). "
                    f"Please increase bucket_size_mb (currently {self.bucket_size_mb} MB)."
                )

                # Absolute offset in the double buffer
                buf_base = (gidx % 2) * self.bucket_size
                tensor_offset = buf_base + offset

                bucket_meta.append(
                    {
                        "name": name,
                        "shape": weight.shape,
                        "dtype": weight.dtype,
                        "offset": tensor_offset,
                    }
                )
                self.buffer[tensor_offset : tensor_offset + weight_nbytes].copy_(
                    weight.view(-1).view(torch.uint8), non_blocking=True
                )
                offset += weight_nbytes

            # Send the last bucket
            if bucket_meta:
                torch.cuda.synchronize()
                self._send_bucket(bucket_meta)

            # Release signal (first None) → all receivers release the buffer
            self._send_none_to_all()

            # Post-hook signal (second None) → all receivers run post_hook
            self._send_none_to_all()

        finally:
            self._cleanup()

    def _init_sockets(self) -> None:
        """Create N REQ sockets, one per TP rank, each bound to its path."""
        for path in self.socket_paths:
            if path.startswith("ipc://"):
                ipc_path = path[len("ipc://") :]
                try:
                    os.remove(ipc_path)
                except OSError:
                    pass
            sock = self.zmq_context.socket(zmq.REQ)
            sock.bind(path)
            self.sockets.append(sock)

    def _init_buffer(self) -> None:
        """Allocate the double buffer and send its IPC handle to all sockets."""
        # Double buffer: bucket_size * 2 for double-buffering
        self.buffer = torch.empty(
            self.bucket_size * 2,
            dtype=torch.uint8,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        handle = reduce_tensor(self.buffer)

        # Send IPC handle to all N sockets, collect acks
        for sock in self.sockets:
            sock.send_pyobj(handle)
        for sock in self.sockets:
            ack = sock.recv()
            if ack != b"":
                # Receiver sends error string on failure
                raise RuntimeError(
                    f"CkptEngineWeightSender: receiver handshake failed: {ack.decode('utf-8', errors='replace')}"
                )

    def _send_bucket(self, metadata: List[Dict[str, Any]]) -> None:
        """Send bucket metadata (list[dict]) to all N sockets, collect acks."""
        # Send to all sockets
        for sock in self.sockets:
            sock.send_pyobj(metadata)
        # Collect acks from all sockets
        for i, sock in enumerate(self.sockets):
            ack = sock.recv()
            if ack != b"":
                raise RuntimeError(
                    f"CkptEngineWeightSender: receiver {i} bucket load failed: {ack.decode('utf-8', errors='replace')}"
                )

    def _send_none_to_all(self) -> None:
        """Send None to all N sockets, collect acks."""
        for sock in self.sockets:
            sock.send_pyobj(None)
        for sock in self.sockets:
            sock.recv()

    def _cleanup(self) -> None:
        """Close all sockets and release the buffer."""
        for sock in self.sockets:
            try:
                sock.close(linger=0)
            except Exception:
                pass
        self.sockets = []

        for path in self.socket_paths:
            if path.startswith("ipc://"):
                ipc_path = path[len("ipc://") :]
                try:
                    os.remove(ipc_path)
                except OSError:
                    pass

        del self.buffer
        self.buffer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()


__all__ = ["CkptEngineWeightSender"]
