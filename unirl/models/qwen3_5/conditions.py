"""Qwen3_5ARConditions — typed conditions container for the Qwen3.5 AR stage.

Mirror of :class:`unirl.models.qwen_vl.QwenVLARConditions` plus a
``video_grid_thw`` slot (Qwen3.5 adds first-class video tokens).
``pixel_values`` / ``image_grid_thw`` / ``video_grid_thw`` are per-sample
lists (FieldKind.CONCAT) so multi-worker concatenation preserves every
sample's media tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import TextTokenCondition


@dataclass
class Qwen3_5ARConditions(Batch):
    """Conditions for Qwen3.5 autoregressive generation (text + image + video)."""

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    pixel_values: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    image_grid_thw: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    video_grid_thw: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3_5ARConditions":
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                f"Qwen3_5ARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got "
                f"{type(prompt).__name__ if prompt is not None else 'None'}"
            )
        return cls(
            prompt=prompt,
            pixel_values=d.get("pixel_values"),
            image_grid_thw=d.get("image_grid_thw"),
            video_grid_thw=d.get("video_grid_thw"),
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.prompt is None:
            raise ValueError("Qwen3_5ARConditions.to_dict: prompt field is None")
        return {
            "prompt": self.prompt,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
            "video_grid_thw": self.video_grid_thw,
        }


__all__ = ["Qwen3_5ARConditions"]
