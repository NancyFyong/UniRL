# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Qwen3_5ChatTemplateStage — ``Texts (+ images) -> Qwen3_5ARConditions``.

Applies the bundle processor's chat template (with
``add_generation_prompt=True`` and ``enable_thinking``) so the AR stage
starts from the canonical assistant-turn prefix. Mirrors
:class:`unirl.models.qwen_vl.QwenVLChatTemplateStage`'s per-sample
processor call (to extract ``pixel_values`` / ``image_grid_thw`` /
``video_grid_thw``) plus Qwen3's ``enable_thinking`` switch.
"""

from __future__ import annotations

from typing import List, Optional

import PIL.Image
import torch

from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts

from .bundle import Qwen3_5Bundle
from .conditions import Qwen3_5ARConditions


class Qwen3_5ChatTemplateStage:
    """Apply the Qwen3.5 chat template, right-pad in batch, return AR conditions."""

    def __init__(
        self,
        bundle: Qwen3_5Bundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        enable_thinking: bool = False,
        pad_to_max_length: bool = False,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        # MUST agree with the rollout engine's chat_template_kwargs.enable_thinking
        # or train/rollout prompts diverge and the importance ratio breaks.
        self.enable_thinking = bool(enable_thinking)
        # When True, pad every prompt to max_prompt_length (v2 DP trainer shard
        # concatenation requires a single shared seq len across shards).
        self.pad_to_max_length = bool(pad_to_max_length)

    def embed(
        self,
        texts: Texts,
        images: Optional[List[Optional[PIL.Image.Image]]] = None,
    ) -> Qwen3_5ARConditions:
        processor = self.bundle.processor
        device = self.bundle.device
        dtype = self.bundle.dtype
        batch_size = len(texts.texts)

        per_sample_inputs = []
        for i, text in enumerate(texts.texts):
            content: list = []
            if images is not None and i < len(images) and images[i] is not None:
                content.append({"type": "image", "image": images[i]})
            content.append({"type": "text", "text": text})

            messages: list = []
            if self.system_instruction is not None:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": content})

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            per_sample_inputs.append(inputs)

        if self.pad_to_max_length:
            max_len = self.max_prompt_length
        else:
            max_len = min(
                max(inp["input_ids"].shape[-1] for inp in per_sample_inputs),
                self.max_prompt_length,
            )

        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "Qwen3_5ChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "Qwen3_5Bundle.from_config sets pad_token=eos_token when absent."
            )

        input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=device
        )

        for i, inp in enumerate(per_sample_inputs):
            ids = inp["input_ids"].squeeze(0)
            L = min(int(ids.shape[0]), max_len)
            input_ids[i, :L] = ids[:L].to(device)
            mask = inp["attention_mask"].squeeze(0)
            attention_mask[i, :L] = mask[:L].to(device)

        # Per-sample lists for media tensors (FieldKind.CONCAT-safe).
        pixel_values: List[Optional[torch.Tensor]] = []
        image_grid_thw: List[Optional[torch.Tensor]] = []
        video_grid_thw: List[Optional[torch.Tensor]] = []
        for inp in per_sample_inputs:
            pv = inp.get("pixel_values")
            igt = inp.get("image_grid_thw")
            vgt = inp.get("video_grid_thw")
            pixel_values.append(
                pv.to(device=device, dtype=dtype) if pv is not None else None
            )
            image_grid_thw.append(igt.to(device=device) if igt is not None else None)
            video_grid_thw.append(vgt.to(device=device) if vgt is not None else None)

        has_img = any(p is not None for p in pixel_values)
        has_vid = any(v is not None for v in video_grid_thw)

        return Qwen3_5ARConditions(
            prompt=TextTokenCondition(
                input_ids=input_ids, attention_mask=attention_mask
            ),
            pixel_values=pixel_values if has_img else None,
            image_grid_thw=image_grid_thw if has_img else None,
            video_grid_thw=video_grid_thw if has_vid else None,
        )


__all__ = ["Qwen3_5ChatTemplateStage"]
