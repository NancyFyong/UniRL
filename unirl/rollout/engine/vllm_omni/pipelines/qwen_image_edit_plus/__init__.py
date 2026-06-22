"""Qwen-Image-Edit-Plus-specific subclass for the vLLM-Omni rollout engine.

Importing this package's ``RLQwenImageEditPlusPipeline`` will fail outside a
vLLM-Omni-equipped environment because the parent class lives in
``vllm_omni``; that's intentional — this module is only meant to be
imported inside vLLM-Omni's worker subprocess via
``custom_pipeline_args``.
"""

__all__: list[str] = []
