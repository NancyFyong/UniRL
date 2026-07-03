# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Qwen3.5 VL (dense + MoE, hybrid Gated Delta Net attention) AR model package.

AR-only VL pipeline: text (+ images) in, text out. Supports GRPO training via
``Qwen3_5ARStage.autoregress`` + ``Qwen3_5ARStage.replay``.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.qwen3_5.ar import (
    Qwen3_5ARParams,
    Qwen3_5ARStage,
    Qwen3_5ARStep,
)
from unirl.models.qwen3_5.bundle import Qwen3_5Bundle
from unirl.models.qwen3_5.chat_template import Qwen3_5ChatTemplateStage
from unirl.models.qwen3_5.conditions import Qwen3_5ARConditions
from unirl.models.qwen3_5.config import Qwen3_5PipelineConfig
from unirl.models.qwen3_5.pipeline import Qwen3_5Pipeline

__all__ = [
    "Qwen3_5ARConditions",
    "Qwen3_5ARParams",
    "Qwen3_5ARStage",
    "Qwen3_5ARStep",
    "Qwen3_5Bundle",
    "Qwen3_5ChatTemplateStage",
    "Qwen3_5Pipeline",
    "Qwen3_5PipelineConfig",
]
