"""Checkpoint-engine ZMQ + CUDA IPC sender compatible with SGLang ``update_weights_from_ipc``."""

from __future__ import annotations

import gc
import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

_PARAMETER_ALIGNMENT = 256
logger = logging.getLogger(__name__)


class CoordinatedWeightSyncError(RuntimeError):
    """A transport failure already observed by every training rank."""


class CkptEngineWeightSender:
    """Send full weights over one REQ socket per colocated SGLang scheduler."""

    def __init__(
        self,
        socket_path: str,
        bucket_size_mb: int,
        timeout_s: int,
    ) -> None:
        self.socket_path = socket_path
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = self.bucket_size_mb << 20
        self.timeout_ms = timeout_s * 1000
        if self.bucket_size <= 0:
            raise ValueError(f"bucket_size_mb must be positive; got {bucket_size_mb}")

        self.zmq_context = zmq.Context.instance()
        self.socket: Optional[zmq.Socket] = None
        self._can_send = False
        self.buffer = None
        self._handle = None
        self._abort_sent = False
        self._deadline: Optional[float] = None

    def prepare(self) -> None:
        """Allocate the CUDA IPC buffer; sockets are created later on the sender thread."""
        if self.buffer is not None or self._handle is not None:
            raise RuntimeError("CkptEngineWeightSender is already prepared")
        # Every payload waits for its receiver ACK before this buffer is reused,
        # so a second half cannot overlap useful work and only increases peak VRAM.
        self.buffer = torch.empty(
            self.bucket_size,
            dtype=torch.uint8,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._handle = reduce_tensor(self.buffer)

    def send_weights(
        self,
        weights: Iterator[Tuple[str, "object"]],
        consensus: Callable[[Optional[BaseException], str], None],
    ) -> None:
        """Send ``(name, tensor)`` pairs to every supplied receiver."""
        try:
            self._deadline = time.monotonic() + self.timeout_ms / 1000
            if self.buffer is None or self._handle is None:
                raise RuntimeError("CkptEngineWeightSender.prepare() must succeed before send_weights()")
            # Create/bind sockets in THIS thread (see ``prepare`` docstring):
            # pyzmq sockets must live and die on the thread that uses them.
            self._init_socket()
            self._exchange(self._handshake, consensus, "handshake")

            offset = 0
            bucket_index = 0
            bucket_meta: List[Dict[str, Any]] = []
            tensor_index = 0

            while True:
                item = None
                exhausted = False
                stage_error: Optional[BaseException] = None
                try:
                    item = next(weights)
                except StopIteration:
                    exhausted = True
                except BaseException as exc:
                    try:
                        self._abort_training_group()
                    except BaseException as abort_exc:
                        raise CoordinatedWeightSyncError(
                            "CkptEngineWeightSender: tensor materialization failed and "
                            "the training process group could not be aborted"
                        ) from BaseExceptionGroup(
                            "materialization and process-group abort failures",
                            [exc, abort_exc],
                        )
                    raise CoordinatedWeightSyncError(
                        "CkptEngineWeightSender: tensor materialization failed; training process group aborted"
                    ) from exc

                try:
                    if not exhausted:
                        name, weight = item
                        weight_nbytes = weight.nbytes
                        aligned_offset = self._align_offset(offset)

                        # Flush current bucket if this tensor doesn't fit.
                        if aligned_offset + weight_nbytes > self.bucket_size and bucket_meta:
                            torch.cuda.current_stream().synchronize()
                            self._exchange(
                                lambda: self._send_payload(bucket_meta, "bucket load"),
                                consensus,
                                f"bucket-{bucket_index}",
                            )
                            bucket_index += 1
                            offset = 0
                            bucket_meta = []
                            aligned_offset = 0

                        # ``raise`` (not ``assert``): the check must survive ``python -O``.
                        if aligned_offset + weight_nbytes > self.bucket_size:
                            raise ValueError(
                                f"Weight {name}({weight.shape}, {weight.dtype}) is too large "
                                f"to fit in the bucket ({weight_nbytes} > {self.bucket_size}). "
                                f"Please increase bucket_size_mb (currently {self.bucket_size_mb} MB)."
                            )

                        bucket_meta.append(
                            {
                                "name": name,
                                "shape": weight.shape,
                                "dtype": weight.dtype,
                                "offset": aligned_offset,
                            }
                        )
                        payload = weight.detach().contiguous().view(-1).view(torch.uint8)
                        self.buffer[aligned_offset : aligned_offset + weight_nbytes].copy_(payload)
                        offset = aligned_offset + weight_nbytes
                except CoordinatedWeightSyncError:
                    raise
                except BaseException as exc:
                    stage_error = exc

                # No rank may advance to the next DTensor/FSDP materialization
                # until every rank has staged this tensor or observed exhaustion.
                phase = f"tensor-{tensor_index}-done" if exhausted else f"tensor-{tensor_index}-staged"
                consensus(stage_error, phase)
                if stage_error is not None:
                    raise stage_error
                if exhausted:
                    break

                item = None
                weight = None
                payload = None
                tensor_index += 1

            # Send the last bucket
            if bucket_meta:
                torch.cuda.current_stream().synchronize()
                self._exchange(
                    lambda: self._send_payload(bucket_meta, "bucket load"),
                    consensus,
                    f"bucket-{bucket_index}",
                )

            # Match checkpoint-engine's lifecycle: release both sides of the IPC
            # allocation before running a potentially memory-heavy post-hook.
            self._exchange(
                lambda: self._send_payload(None, "receiver finalization"),
                consensus,
                "release",
            )
            self._release_buffer()
            self._exchange(
                lambda: self._send_payload(None, "receiver finalization"),
                consensus,
                "post-hook",
            )

        except BaseException as exc:
            self._abort_receiver(exc)
            raise
        finally:
            close = getattr(weights, "close", None)
            if callable(close):
                close()
            weight = None
            self._cleanup()

    @staticmethod
    def _align_offset(offset: int) -> int:
        """Align tensor starts to checkpoint-engine's 256-byte boundary."""
        return (offset + _PARAMETER_ALIGNMENT - 1) // _PARAMETER_ALIGNMENT * _PARAMETER_ALIGNMENT

    @staticmethod
    def _abort_training_group() -> None:
        """Fail-stop a rank-local materialization error before peers hang."""
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() == 1:
            return

        world = dist.group.WORLD
        abort = getattr(world, "abort", None)
        if not callable(abort):
            raise RuntimeError("default ProcessGroup has no abort() capability")
        abort()

    @staticmethod
    def _exchange(
        operation: Callable[[], None],
        consensus: Callable[[Optional[BaseException], str], None],
        phase: str,
    ) -> None:
        error = None
        try:
            operation()
        except BaseException as exc:
            error = exc
        consensus(error, phase)
        if error is not None:
            raise error

    def _init_socket(self) -> None:
        """Bind this rank's REQ socket."""
        sock = self.zmq_context.socket(zmq.REQ)
        try:
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(self.socket_path)
        except BaseException:
            sock.close(linger=0)
            raise
        self.socket = sock
        self._can_send = True

    def _require_socket(self) -> zmq.Socket:
        """Return the bound sender socket."""
        if self.socket is None:
            raise RuntimeError("checkpoint-engine sender socket is not initialized")
        return self.socket

    def _handshake(self) -> None:
        """Send the prepared IPC handle to every receiver."""
        sock = self._require_socket()
        self._apply_remaining_timeout(sock)
        sock.send_pyobj(self._handle)
        self._can_send = False
        self._apply_remaining_timeout(sock)
        ack = sock.recv()
        self._can_send = True
        if ack != b"":
            error = ack.decode("utf-8", errors="replace")
            # The worker's handshake error path waits for one raw ACK before
            # raising and closing; it has not entered the payload state machine.
            sock.send(b"")
            self._can_send = False
            raise RuntimeError(f"CkptEngineWeightSender: receiver handshake failed: {error}")

    def _send_payload(self, payload: Any, operation: str) -> None:
        """Send one protocol payload and collect the receiver's ACK."""
        sock = self._require_socket()
        self._apply_remaining_timeout(sock)
        sock.send_pyobj(payload)
        self._can_send = False
        self._apply_remaining_timeout(sock)
        ack = sock.recv()
        self._can_send = True
        if ack != b"":
            error = ack.decode("utf-8", errors="replace")
            raise RuntimeError(f"CkptEngineWeightSender: {operation} failed: {error}")

    def _apply_remaining_timeout(self, sock: zmq.Socket) -> None:
        """Apply one absolute transfer deadline to every socket operation."""
        if self._deadline is None:
            raise RuntimeError("transfer deadline is not initialized")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("CkptEngineWeightSender: transfer deadline exceeded")
        timeout_ms = max(1, int(remaining * 1000))
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)

    def _abort_receiver(self, error: BaseException) -> None:
        """Release the worker after it entered checkpoint-engine's payload loop."""
        if self.socket is None:
            return
        abort = RuntimeError(f"CkptEngineWeightSender aborted: {error}")
        if not self._can_send:
            logger.error(
                "Cannot send checkpoint-engine abort while awaiting an ACK; the orchestrator must terminate SGLang"
            )
            return
        try:
            # checkpoint_engine.worker raises this payload without replying.
            self.socket.send_pyobj(abort)
            self._can_send = False
            self._abort_sent = True
        except Exception:
            logger.exception("Failed to send checkpoint-engine abort")

    def close(self) -> None:
        """Release a prepared sender that did not enter ``send_weights``."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Close the socket and release the buffer."""
        socket = self.socket
        self.socket = None
        if socket is not None:
            try:
                socket.close(linger=5000 if self._abort_sent else 0)
            except Exception:
                logger.exception("Failed to close checkpoint-engine sender socket")
        self._can_send = False

        self._release_buffer()

    def _release_buffer(self) -> None:
        """Drop producer IPC storage after the receiver's release ACK."""
        if self.buffer is None and self._handle is None:
            return
        self.buffer = None
        self._handle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()


__all__ = ["CkptEngineWeightSender", "CoordinatedWeightSyncError"]
