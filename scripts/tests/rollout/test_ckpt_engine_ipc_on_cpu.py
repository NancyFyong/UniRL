"""Tier 0 smoke tests — CkptEngineIPCWeightSync + CkptEngineWeightSender, no GPU.

Validates the checkpoint_engine IPC weight-sync logic without a live SGLang
server or CUDA IPC by driving the pure logic:

  - ``CkptEngineWeightSender`` creates N REQ sockets, sends the IPC handle to
    all N, sends bucket metadata (``list[dict]`` with absolute offsets) to all
    N, and terminates with two ``None`` signals (release + post_hook).
  - ``CkptEngineIPCWeightSync.sync`` only pushes from ``tp_rank==0`` (others
    drain the all-gather generator in lockstep but skip the push), constructs
    the correct ``zmq_handles`` dict, and assigns threads based on backend type
    (NativeBackend: receiver in main thread; HTTPBackend: receiver in daemon
    thread).

The transports are exercised with lightweight fakes for the ZMQ context,
rollout sibling, and the FSDP backend, so no GPU / ZMQ / CUDA IPC is needed.

Run:  pytest scripts/tests/rollout/test_ckpt_engine_ipc_on_cpu.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from unirl.distributed.group.remote import RankInfo
from unirl.distributed.weight_sync.full.ckpt_engine_ipc import CkptEngineIPCWeightSync

from ..conftest import FakeBackend


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@dataclass
class _FakeTensor:
    """Minimal tensor stand-in with ``shape``, ``dtype``, ``nbytes``."""

    shape: tuple
    dtype: Any
    nbytes: int

    def view(self, *args, **kwargs):
        return self


class _FakeZMQSocket:
    """Recording ZMQ socket that returns ``b""`` for all recv() calls."""

    def __init__(self):
        self.sent: List[Any] = []
        self.closed = False
        self.bound_path: Optional[str] = None

    def bind(self, path: str) -> None:
        self.bound_path = path

    def send_pyobj(self, obj: Any) -> None:
        self.sent.append(obj)

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self) -> bytes:
        return b""

    def close(self, linger: int = 0) -> None:
        self.closed = True


class _FakeZMQContext:
    """Returns one ``_FakeZMQSocket`` per ``socket()`` call."""

    def __init__(self):
        self.sockets: List[_FakeZMQSocket] = []

    def socket(self, socket_type: int) -> _FakeZMQSocket:
        sock = _FakeZMQSocket()
        self.sockets.append(sock)
        return sock


class NativeBackend:
    pass


class HTTPBackend:
    pass


class _FakeCkptRollout:
    """Rollout sibling that records ``update_weights_from_ipc`` calls."""

    def __init__(self, tp_size: int = 1, tp_device_ids: Optional[List[int]] = None,
                 backend_name: str = "NativeBackend"):
        self._tp_size = tp_size
        self._tp_device_ids = tp_device_ids
        if backend_name == "NativeBackend":
            self._backend = NativeBackend()
        else:
            self._backend = HTTPBackend()
        self.receiver_calls: List[Dict] = []

    def update_weights_from_ipc(self, **kwargs):
        self.receiver_calls.append(dict(kwargs))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_ckpt_sync(rollout: _FakeCkptRollout, rank_info: Optional[RankInfo] = None):
    """Build a ``CkptEngineIPCWeightSync`` via ``__new__`` bypass."""
    sync = CkptEngineIPCWeightSync.__new__(CkptEngineIPCWeightSync)
    sync._backend = FakeBackend()
    sync._bucket_bytes = 16 << 20  # 16 MB
    sync._flush_cache = True
    sync._lora_merged = False
    sync._adapter_name = "default"
    sync._name_remap = []
    sync._track_prefix = ""
    sync._wire_dtype = None
    sync.weight_version = 0
    sync._rollout = rollout
    sync.rank_info = rank_info or RankInfo(rank=0)
    return sync


# --------------------------------------------------------------------------- #
# CkptEngineWeightSender unit tests
# --------------------------------------------------------------------------- #


def test_ckpt_engine_sender_init():
    """Constructor stores zmq_handles, computes bucket_size and n_sockets."""
    from unirl.distributed.weight_sync.transfer.ckpt_engine_transfer import (
        CkptEngineWeightSender,
    )

    handles = {
        "GPU-uuid-0": "ipc:///tmp/sock0.sock",
        "GPU-uuid-1": "ipc:///tmp/sock1.sock",
    }
    sender = CkptEngineWeightSender(zmq_handles=handles, bucket_size_mb=512)
    assert sender.n_sockets == 2
    assert sender.bucket_size_mb == 512
    assert sender.bucket_size == 512 << 20
    assert len(sender.zmq_handles) == 2


def test_ckpt_engine_sender_multi_socket_metadata(monkeypatch):
    """Sender creates N REQ sockets, sends IPC handle + metadata + two Nones to all."""
    from unirl.distributed.weight_sync.transfer import ckpt_engine_transfer as mod

    fake_ctx = _FakeZMQContext()

    # Patch zmq.Context.instance to return our fake
    monkeypatch.setattr(mod.zmq, "Context", type("_ZMQMod", (), {"instance": staticmethod(lambda: fake_ctx)}))

    # Patch torch.cuda and torch.empty
    _fake_cuda = type("_FakeCuda", (), {
        "current_device": staticmethod(lambda: 0),
        "synchronize": staticmethod(lambda: None),
        "ipc_collect": staticmethod(lambda: None),
        "empty_cache": staticmethod(lambda: None),
        "is_available": staticmethod(lambda: True),
    })()
    monkeypatch.setattr(mod.torch, "cuda", _fake_cuda)

    class _FakeBuf:
        def copy_(self, *a, **kw): pass
        def __getitem__(self, key): return self
    monkeypatch.setattr(mod.torch, "empty", lambda *a, **kw: _FakeBuf())

    # reduce_tensor returns a sentinel tuple
    monkeypatch.setattr(mod, "reduce_tensor", lambda buf: ("fake_handle", ()))

    handles = {"GPU-0": "ipc:///tmp/s0.sock", "GPU-1": "ipc:///tmp/s1.sock"}
    sender = mod.CkptEngineWeightSender(zmq_handles=handles, bucket_size_mb=16)

    # Feed one small tensor
    tensor = _FakeTensor(shape=(4,), dtype=type("DType", (), {"itemsize": 2})(), nbytes=8)
    asyncio.run(sender.async_send_weights([("w0", tensor)]))

    # 2 sockets created
    assert len(fake_ctx.sockets) == 2
    for sock in fake_ctx.sockets:
        # Each socket received: IPC handle, 1 bucket metadata, None (release), None (post_hook)
        assert len(sock.sent) == 4
        assert sock.sent[0] == ("fake_handle", ())  # IPC handle
        assert isinstance(sock.sent[1], list)  # bucket metadata (list[dict])
        assert sock.sent[1][0]["name"] == "w0"
        assert sock.sent[2] is None  # release signal
        assert sock.sent[3] is None  # post_hook signal


def test_ckpt_engine_sender_double_buffer_offsets(monkeypatch):
    """Absolute offsets alternate between buf_base=0 and buf_base=bucket_size."""
    from unirl.distributed.weight_sync.transfer import ckpt_engine_transfer as mod

    fake_ctx = _FakeZMQContext()
    monkeypatch.setattr(mod.zmq, "Context", type("_ZMQMod", (), {"instance": staticmethod(lambda: fake_ctx)}))

    _fake_cuda = type("_FakeCuda", (), {
        "current_device": staticmethod(lambda: 0),
        "synchronize": staticmethod(lambda: None),
        "ipc_collect": staticmethod(lambda: None),
        "empty_cache": staticmethod(lambda: None),
        "is_available": staticmethod(lambda: True),
    })()
    monkeypatch.setattr(mod.torch, "cuda", _fake_cuda)

    class _FakeBuf:
        def copy_(self, *a, **kw): pass
        def __getitem__(self, key): return self
    monkeypatch.setattr(mod.torch, "empty", lambda *a, **kw: _FakeBuf())
    monkeypatch.setattr(mod, "reduce_tensor", lambda buf: ("handle", ()))

    # bucket_size = 16 MB, each tensor = 4 MB → 4 per bucket
    handles = {"GPU-0": "ipc:///tmp/s0.sock"}
    sender = mod.CkptEngineWeightSender(zmq_handles=handles, bucket_size_mb=16)

    tensor_size = 4 * 1024 * 1024  # 4 MB
    dtype = type("DType", (), {"itemsize": 1})()
    tensors = [(f"w{i}", _FakeTensor(shape=(tensor_size,), dtype=dtype, nbytes=tensor_size)) for i in range(13)]

    asyncio.run(sender.async_send_weights(tensors))

    sock = fake_ctx.sockets[0]
    # IPC handle + 4 buckets + None + None = 7 messages
    assert len(sock.sent) == 7

    # First bucket (gidx=0, buf_base=0)
    bucket0 = sock.sent[1]
    assert isinstance(bucket0, list)
    assert bucket0[0]["offset"] < 16 << 20  # first half

    # Second bucket (gidx=1, buf_base=bucket_size)
    bucket1 = sock.sent[2]
    assert isinstance(bucket1, list)
    assert bucket1[0]["offset"] >= 16 << 20  # second half

    # Third bucket (gidx=2, buf_base=0 again)
    bucket2 = sock.sent[3]
    assert isinstance(bucket2, list)
    assert bucket2[0]["offset"] < 16 << 20  # back to first half


def test_ckpt_engine_sender_asserts_large_tensor(monkeypatch):
    """Assertion fires when a tensor exceeds bucket_size."""
    from unirl.distributed.weight_sync.transfer import ckpt_engine_transfer as mod

    fake_ctx = _FakeZMQContext()
    monkeypatch.setattr(mod.zmq, "Context", type("_ZMQMod", (), {"instance": staticmethod(lambda: fake_ctx)}))

    _fake_cuda = type("_FakeCuda", (), {
        "current_device": staticmethod(lambda: 0),
        "synchronize": staticmethod(lambda: None),
        "ipc_collect": staticmethod(lambda: None),
        "empty_cache": staticmethod(lambda: None),
        "is_available": staticmethod(lambda: True),
    })()
    monkeypatch.setattr(mod.torch, "cuda", _fake_cuda)

    class _FakeBuf:
        def copy_(self, *a, **kw): pass
        def __getitem__(self, key): return self
    monkeypatch.setattr(mod.torch, "empty", lambda *a, **kw: _FakeBuf())
    monkeypatch.setattr(mod, "reduce_tensor", lambda buf: ("handle", ()))

    handles = {"GPU-0": "ipc:///tmp/s0.sock"}
    sender = mod.CkptEngineWeightSender(zmq_handles=handles, bucket_size_mb=1)  # 1 MB bucket

    # Tensor larger than 1 MB
    big_tensor = _FakeTensor(shape=(2 << 20,), dtype=type("D", (), {"itemsize": 1})(), nbytes=2 << 20)
    with pytest.raises(AssertionError, match="too large"):
        asyncio.run(sender.async_send_weights([("big", big_tensor)]))


# --------------------------------------------------------------------------- #
# CkptEngineIPCWeightSync unit tests
# --------------------------------------------------------------------------- #


def test_ckpt_ipc_sync_tp_zero_pushes(monkeypatch):
    """tp_rank==0: receiver spawned, sender called, weight_version incremented."""
    rollout = _FakeCkptRollout(tp_size=2, tp_device_ids=[0, 1])
    sync = _make_ckpt_sync(rollout, RankInfo(rank=0, tp_rank=0, tp_size=2))

    sender_calls: List[Dict] = []

    def _fake_run_sender(zmq_handles):
        sender_calls.append({"zmq_handles": zmq_handles})

    monkeypatch.setattr(sync, "_run_sender", _fake_run_sender)
    monkeypatch.setattr(sync, "_iter_full_tensors", lambda: iter([("w0", _FakeTensor((4,), type("D", (), {"itemsize": 2})(), 8))]))
    monkeypatch.setattr(sync, "_build_zmq_handles", lambda tp_size, base_gpu_id: {f"GPU-{i}": f"ipc:///tmp/s{i}.sock" for i in range(tp_size)})
    monkeypatch.setattr(sync, "_is_native_backend", lambda: False)

    sync.sync()

    assert len(rollout.receiver_calls) == 1
    assert "zmq_handles" in rollout.receiver_calls[0]
    assert len(rollout.receiver_calls[0]["zmq_handles"]) == 2
    assert len(sender_calls) == 1
    assert sync.weight_version == 1


def test_ckpt_ipc_sync_non_tp_zero_skips_push(monkeypatch):
    """tp_rank>0: all-gather runs (drains generator) but no push, weight_version unchanged."""
    rollout = _FakeCkptRollout(tp_size=2, tp_device_ids=[0, 1])
    sync = _make_ckpt_sync(rollout, RankInfo(rank=1, tp_rank=1, tp_size=2))

    sender_calls: List[Dict] = []
    monkeypatch.setattr(sync, "_run_sender", lambda zmq_handles: sender_calls.append(zmq_handles))

    drained: List[str] = []
    def _drain_gen():
        drained.append("walk")
        yield "w0", _FakeTensor((4,), type("D", (), {"itemsize": 2})(), 8)
    monkeypatch.setattr(sync, "_iter_full_tensors", _drain_gen)

    sync.sync()

    assert drained == ["walk"]  # generator was drained (lockstep all-gather)
    assert len(rollout.receiver_calls) == 0  # no push
    assert len(sender_calls) == 0
    assert sync.weight_version == 0  # unchanged


def test_ckpt_ipc_sync_native_backend_thread_assignment(monkeypatch):
    """NativeBackend: receiver in main thread, sender in daemon thread."""
    rollout = _FakeCkptRollout(tp_size=1, tp_device_ids=[0], backend_name="NativeBackend")
    sync = _make_ckpt_sync(rollout, RankInfo(rank=0, tp_rank=0, tp_size=1))

    sender_calls: List[Dict] = []
    monkeypatch.setattr(sync, "_run_sender", lambda zmq_handles: sender_calls.append(zmq_handles))
    monkeypatch.setattr(sync, "_iter_full_tensors", lambda: iter([("w0", _FakeTensor((4,), type("D", (), {"itemsize": 2})(), 8))]))
    monkeypatch.setattr(sync, "_build_zmq_handles", lambda tp_size, base_gpu_id: {"GPU-0": "ipc:///tmp/s0.sock"})
    monkeypatch.setattr(sync, "_is_native_backend", lambda: True)

    sync.sync()

    assert len(rollout.receiver_calls) == 1
    assert len(sender_calls) == 1
    assert sync.weight_version == 1


def test_ckpt_ipc_sync_http_backend_thread_assignment(monkeypatch):
    """HTTPBackend: receiver in daemon thread, sender in main thread."""
    rollout = _FakeCkptRollout(tp_size=1, tp_device_ids=[0], backend_name="HTTPBackend")
    sync = _make_ckpt_sync(rollout, RankInfo(rank=0, tp_rank=0, tp_size=1))

    sender_calls: List[Dict] = []
    monkeypatch.setattr(sync, "_run_sender", lambda zmq_handles: sender_calls.append(zmq_handles))
    monkeypatch.setattr(sync, "_iter_full_tensors", lambda: iter([("w0", _FakeTensor((4,), type("D", (), {"itemsize": 2})(), 8))]))
    monkeypatch.setattr(sync, "_build_zmq_handles", lambda tp_size, base_gpu_id: {"GPU-0": "ipc:///tmp/s0.sock"})
    monkeypatch.setattr(sync, "_is_native_backend", lambda: False)

    sync.sync()

    assert len(rollout.receiver_calls) == 1
    assert len(sender_calls) == 1
    assert sync.weight_version == 1


def test_ckpt_ipc_build_zmq_handles(monkeypatch):
    """_build_zmq_handles constructs correct {uuid: path} dict for N TP ranks."""
    rollout = _FakeCkptRollout(tp_size=3, tp_device_ids=[2, 3, 4])
    sync = _make_ckpt_sync(rollout)

    class _FakeProps:
        def __init__(self, uuid: str):
            self.uuid = uuid

    props_map = {2: _FakeProps("uuid-2"), 3: _FakeProps("uuid-3"), 4: _FakeProps("uuid-4")}
    import unirl.distributed.weight_sync.full.ckpt_engine_ipc as cie
    monkeypatch.setattr(cie.torch.cuda, "get_device_properties", lambda gpu_id: props_map[gpu_id])

    handles = sync._build_zmq_handles(tp_size=3, base_gpu_id=2)

    assert len(handles) == 3
    assert "GPU-uuid-2" in handles
    assert "GPU-uuid-3" in handles
    assert "GPU-uuid-4" in handles
    assert all(v.startswith("ipc:///tmp/") for v in handles.values())


def test_ckpt_ipc_get_tp_size_and_base_gpu_id():
    """_get_tp_size and _get_base_gpu_id read from rollout's attributes."""
    rollout = _FakeCkptRollout(tp_size=4, tp_device_ids=[2, 3, 4, 5])
    sync = _make_ckpt_sync(rollout)

    assert sync._get_tp_size() == 4
    assert sync._get_base_gpu_id() == 2

    # Test defaults when attributes are missing
    rollout2 = _FakeCkptRollout(tp_size=0, tp_device_ids=None)
    sync2 = _make_ckpt_sync(rollout2)
    assert sync2._get_tp_size() == 1  # defaults to 1
    assert sync2._get_base_gpu_id() == 0  # defaults to 0


def test_ckpt_ipc_is_native_backend():
    """_is_native_backend detects NativeBackend vs HTTPBackend."""
    rollout_native = _FakeCkptRollout(backend_name="NativeBackend")
    sync_native = _make_ckpt_sync(rollout_native)
    assert sync_native._is_native_backend() is True

    rollout_http = _FakeCkptRollout(backend_name="HTTPBackend")
    sync_http = _make_ckpt_sync(rollout_http)
    assert sync_http._is_native_backend() is False
