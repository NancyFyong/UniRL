"""Full base-weight sync handlers for the v2 trainer."""

from typing import TYPE_CHECKING

from unirl.distributed.weight_sync.full.base import FullWeightSync
from unirl.distributed.weight_sync.full.ipc import IPCWeightSync
from unirl.distributed.weight_sync.full.nccl import NCCLWeightSync
from unirl.distributed.weight_sync.full.tensor import TensorWeightSync

if TYPE_CHECKING:
    from unirl.distributed.weight_sync.full.ckpt_engine_ipc import CkptEngineIPCWeightSync


def __getattr__(name: str):
    """Load the optional checkpoint-engine transport only when requested."""
    if name == "CkptEngineIPCWeightSync":
        from unirl.distributed.weight_sync.full.ckpt_engine_ipc import CkptEngineIPCWeightSync

        return CkptEngineIPCWeightSync
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FullWeightSync",
    "NCCLWeightSync",
    "TensorWeightSync",
    "IPCWeightSync",
    "CkptEngineIPCWeightSync",
]
