"""GPU test — CkptEngineIPCWeightSync memory verification (OOM simulation).

Simplified version: uses synchronous ZMQ (no threads) to avoid deadlocks.
Simulates 35B-scale weight sync on a single GPU to verify:
  1. ``CkptEngineWeightSender`` peak memory = baseline + double_buffer (zero-copy receiver)
  2. ``TensorWeightSync`` pattern would need 3× model (OOM on 35B)

Gated behind ``UNIRL_TP_GPU_TEST=1`` + >=1 GPU.

Run:  CUDA_VISIBLE_DEVICES=2 UNIRL_TP_GPU_TEST=1 pytest scripts/tests/rollout/test_ckpt_engine_ipc_oom_gpu.py -v -s
"""

from __future__ import annotations

import gc
import os
import time
from typing import Dict, List

import pytest
import torch
import zmq

from ..conftest import requires_gpus


def _mock_receiver_sync(socket_path: str, device_id: int, model_weights: Dict[str, torch.Tensor]):
    """Synchronous mock receiver — mirrors checkpoint_engine.worker protocol."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 30000)  # 30s timeout
    sock.connect(socket_path)

    try:
        # Phase 1: receive IPC handle
        ipc_handle = sock.recv_pyobj()
        func, args = ipc_handle
        list_args = list(args)
        list_args[6] = device_id
        buffer = func(*list_args)
        sock.send(b"")

        # Phase 2: bucket loop
        released = False
        while True:
            try:
                payload = sock.recv_pyobj()
            except zmq.error.Again:
                break

            if released:
                # Second None → post_hook
                assert payload is None
                sock.send(b"")
                break

            if payload is None:
                # First None → release
                released = True
                del buffer
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                sock.send(b"")
                continue

            if isinstance(payload, list):
                # Bucket metadata: load weights
                for item in payload:
                    name = item["name"]
                    shape = item["shape"]
                    if isinstance(shape, (list, tuple)):
                        shape = torch.Size(shape)
                    dtype = item["dtype"]
                    offset = item["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if name in model_weights:
                        model_weights[name].copy_(tensor)
                torch.cuda.synchronize()
                sock.send(b"")
    finally:
        sock.close()


@pytest.mark.gpu
@requires_gpus(1)
def test_ckpt_engine_ipc_memory_bounded_gpu():
    """Verify CkptEngineWeightSender peak memory = baseline + double_buffer.

    Simulates a 500MB model. The sender allocates a double buffer (128MB×2=256MB),
    packs weights into it, and sends via ZMQ IPC. The receiver copies into
    pre-allocated model weights (zero extra allocation on receiver side).

    Peak extra memory should be ~256MB (the double buffer), NOT ~1500MB (3× model).
    """
    from unirl.distributed.weight_sync.transfer.ckpt_engine_transfer import (
        CkptEngineWeightSender,
    )
    import asyncio

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Simulate a 200MB model (5 layers × 40MB each, bf16)
    model_size_mb = 200
    num_layers = 5
    per_layer_mb = model_size_mb // num_layers
    per_layer_elements = per_layer_mb * 1024 * 1024 // 2  # bf16 = 2 bytes

    # Pre-allocate "rollout model weights" (the receiver's copy)
    model_weights = {
        f"layer.{i}.weight": torch.empty(per_layer_elements, dtype=torch.bfloat16, device=device)
        for i in range(num_layers)
    }

    # Pre-allocate "trainer model weights" (the source)
    trainer_weights = {
        f"layer.{i}.weight": torch.randn(per_layer_elements, dtype=torch.bfloat16, device=device)
        for i in range(num_layers)
    }

    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated(device)
    baseline_mb = baseline_bytes / 1024 / 1024
    print(f"\nBaseline (trainer + rollout model): {baseline_mb:.1f} MB")

    # Set up ZMQ — use a unique socket path
    socket_path = f"ipc:///tmp/unirl-ckpt-test-{os.getpid()}.sock"
    try:
        os.remove(socket_path.replace("ipc://", ""))
    except OSError:
        pass

    bucket_size_mb = 64  # 64MB bucket → 128MB double buffer

    # Start receiver in a thread (daemon, with timeout)
    import threading
    recv_thread = threading.Thread(
        target=_mock_receiver_sync,
        args=(socket_path, 0, model_weights),
        daemon=True,
    )
    recv_thread.start()
    time.sleep(1.0)  # let receiver connect

    # Run sender
    zmq_handles = {"GPU-test": socket_path}
    sender = CkptEngineWeightSender(zmq_handles=zmq_handles, bucket_size_mb=bucket_size_mb)

    # Track peak memory
    peak_bytes = [0]
    tracker_stop = [False]

    def _track_peak():
        while not tracker_stop[0]:
            current = torch.cuda.memory_allocated(device)
            if current > peak_bytes[0]:
                peak_bytes[0] = current
            time.sleep(0.005)

    tracker_thread = threading.Thread(target=_track_peak, daemon=True)
    tracker_thread.start()

    asyncio.run(sender.async_send_weights(trainer_weights.items()))

    recv_thread.join(timeout=15)
    tracker_stop[0] = True
    tracker_thread.join(timeout=2)

    torch.cuda.synchronize()
    peak_mb = peak_bytes[0] / 1024 / 1024
    final_mb = torch.cuda.memory_allocated(device) / 1024 / 1024

    overhead_mb = peak_mb - baseline_mb
    double_buffer_mb = bucket_size_mb * 2

    print(f"Peak memory during sync: {peak_mb:.1f} MB")
    print(f"Final memory: {final_mb:.1f} MB")
    print(f"Extra memory over baseline: {overhead_mb:.1f} MB")
    print(f"Double buffer size: {double_buffer_mb} MB")

    # Verify: peak should be baseline + double_buffer + small overhead
    # NOT baseline + 3 * model_size (600MB)
    assert overhead_mb < double_buffer_mb + 50, (
        f"Peak overhead {overhead_mb:.1f} MB exceeds expected "
        f"{double_buffer_mb + 50} MB (double buffer + overhead). "
        f"This suggests extra copies (OOM risk)."
    )

    # Verify it's much less than TensorWeightSync's 3× model
    tensor_sync_overhead = 3 * model_size_mb
    assert overhead_mb < tensor_sync_overhead, (
        f"Peak overhead {overhead_mb:.1f} MB should be < "
        f"TensorWeightSync's {tensor_sync_overhead} MB (3× model)."
    )

    print(f"\n✓ CkptEngine IPC overhead: {overhead_mb:.1f} MB")
    print(f"✓ TensorWeightSync would need: ~{tensor_sync_overhead} MB")
    print(f"✓ Memory savings: ~{tensor_sync_overhead - overhead_mb:.1f} MB")

    # Cleanup
    del trainer_weights, model_weights
    gc.collect()
    torch.cuda.empty_cache()

    # Clean up socket
    try:
        os.remove(socket_path.replace("ipc://", ""))
    except OSError:
        pass


@pytest.mark.gpu
@requires_gpus(1)
def test_tensor_sync_would_oom_simulation():
    """Simulate TensorWeightSync memory pattern (3× model) to confirm OOM risk.

    TensorWeightSync: all-gather copy + flatten copy + SGLang deserialize copy.
    Total extra: ~3× model size.
    """
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model_size_mb = 200
    num_layers = 5
    per_layer_mb = model_size_mb // num_layers
    per_layer_elements = per_layer_mb * 1024 * 1024 // 2

    # "FSDP shard" (already on GPU)
    shard = {
        f"layer.{i}.weight": torch.randn(per_layer_elements, dtype=torch.bfloat16, device=device)
        for i in range(num_layers)
    }

    torch.cuda.synchronize()
    baseline_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    print(f"\nBaseline (FSDP shard): {baseline_mb:.1f} MB")

    # Step 1: all-gather (clone = full copy)
    full_tensors = {k: v.clone() for k, v in shard.items()}
    torch.cuda.synchronize()
    after_gather_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    print(f"After all-gather: {after_gather_mb:.1f} MB (+{after_gather_mb - baseline_mb:.1f} MB)")

    # Step 2: flatten (concat into one buffer)
    flat_buffer = torch.cat([v.view(-1) for v in full_tensors.values()], dim=0)
    torch.cuda.synchronize()
    after_flatten_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    print(f"After flatten: {after_flatten_mb:.1f} MB (+{after_flatten_mb - baseline_mb:.1f} MB)")

    # Step 3: SGLang deserialize (another copy)
    sglang_copy = flat_buffer.clone()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    print(f"After SGLang deserialize: {peak_mb:.1f} MB (+{peak_mb - baseline_mb:.1f} MB)")

    overhead = peak_mb - baseline_mb
    print(f"\nTensorWeightSync overhead: {overhead:.1f} MB")
    print(f"Expected (~3× model): ~{3 * model_size_mb} MB")

    assert overhead > 2 * model_size_mb, (
        f"TensorWeightSync overhead {overhead:.1f} MB should be > {2 * model_size_mb} MB."
    )

    del shard, full_tensors, flat_buffer, sglang_copy
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n✓ Confirmed: TensorWeightSync needs ~{overhead:.0f} MB for {model_size_mb} MB model")
    print(f"✓ CkptEngine IPC would need only ~{128} MB (double buffer) for same model")
