"""Qwen-Image-Edit-Plus family: image-edit modality (text+image → image).

Single diffusion stage, TP=1, no AR prelude. Sibling of
:mod:`unirl.rollout.engine.vllm_omni.adapters.qwen_image` (the T2I modality)
with two image-edit deltas:

- **Request side.** Edit-Plus **requires** ``req.primitives['image']: Images``
  (fail-fast if absent — Edit-Plus is edit-only). The input adapter extracts
  PILs via :func:`pil_images_from_req` and injects each into the per-prompt
  dict's ``multi_modal_data.image`` slot. Upstream's
  ``get_qwen_image_edit_plus_pre_process_func`` (auto-registered in
  ``vllm_omni/diffusion/registry.py`` for ``QwenImageEditPlusPipeline``) reads
  that PIL, resizes it to condition_size (384² for the Qwen2.5-VL text encoder)
  + vae_size (1024² for the VAE encoder), and stashes the preprocessed tensors
  into ``prompt["additional_information"]`` keys the EditPlus forward reads.
  The driver never touches image preprocessing — it hands the PIL over and
  lets upstream own the VAE/condition-image pipeline.
- **Response side.** Identical to T2I for the text-capture conditions
  (``text`` always, ``negative_text`` under CFG). The Edit-Plus-specific
  ``image_latent`` condition (the VAE-encoded source-image latent, needed by
  the trainer-side replay's token-concat in
  :meth:`QwenImageEditPlusDiffusionStep.predict_noise`) is captured by the RL
  pipeline subclass (:class:`RLQwenImageEditPlusPipeline`) via a
  ``prepare_latents`` override that stashes the upstream-computed
  ``image_latents`` (the second element of the ``(latents, image_latents)``
  return tuple), unpacks it back to spatial ``[B, 16, H/8, W/8]``, and stamps
  it as ``image_capture`` on ``DiffusionOutput.custom_output``. This adapter
  wraps that capture as an :class:`ImageLatentCondition` and emits it in the
  conditions dict alongside ``text`` / ``negative_text``.

Everything else (CFG semantics, ``true_cfg_scale`` mapping, ragged-pad-concat
of the variable-length Qwen2.5-VL text embeds) is inherited from the T2I
sub-adapters — same checkpoint family, same text encoder.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.qwen_image import (
    QwenImageInputAdapter,
    QwenImageOutputAdapter,
)
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult
from unirl.rollout.engine.vllm_omni.utils import collect_dit_outputs, pil_images_from_req, texts_from_req
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


class QwenImageEditPlusInputAdapter(QwenImageInputAdapter):
    """Edit-Plus request side — injects source-image PIL into each prompt dict.

    Override of :meth:`build_prompts` only; :meth:`build_sampling` is inherited
    from :class:`QwenImageInputAdapter` (the ``true_cfg_scale`` mapping +
    ``max_sequence_length`` pin are unchanged — Edit-Plus uses the same Qwen2.5-VL
    text encoder and the same norm-corrected CFG blend).

    The PIL is handed to upstream verbatim — upstream's
    ``get_qwen_image_edit_plus_pre_process_func`` (selected by the engine via
    the model class registry) resizes it to condition_size + vae_size and
    stashes the preprocessed tensors into
    ``prompt["additional_information"]``. The driver never replicates VAE
    preprocessing.
    """

    def build_prompts(self, req: RolloutReq) -> List[Any]:
        """``{"prompt", "multi_modal_data": {"image": pil}}`` dicts; ``negative_prompt`` ONLY when CFG is armed.

        Edit-Plus **requires** a source image per prompt (fail-fast if absent).
        The CFG arming logic mirrors the T2I sibling: ``negative_prompt`` is
        emitted only when ``guidance_scale > 1.0`` so upstream's
        ``""`` → ``true_cfg 4.0`` default never fires.
        """
        texts = texts_from_req(req)
        n = len(texts.texts)
        pil_images = pil_images_from_req(req, n)
        if not pil_images:
            raise ValueError(
                f"modality={self.modality!r} requires req.primitives['image'] (Edit-Plus is edit-only); got None."
            )
        diff_params = req.sampling_params.get("diffusion")
        if float(diff_params.guidance_scale) > 1.0:
            negative_prompt = str(getattr(diff_params, "negative_prompt", "") or "")
            return [
                {"prompt": text, "negative_prompt": negative_prompt, "multi_modal_data": {"image": pil}}
                for text, pil in zip(texts.texts, pil_images)
            ]
        return [{"prompt": text, "multi_modal_data": {"image": pil}} for text, pil in zip(texts.texts, pil_images)]


class QwenImageEditPlusOutputAdapter(QwenImageOutputAdapter):
    """Edit-Plus response side — T2I text-capture conditions + ``image_latent``.

    Extends :meth:`build_conditions` to also emit ``image_latent`` (an
    :class:`ImageLatentCondition` carrying the VAE-encoded source-image latent
    in spatial ``[B, 16, H/8, W/8]`` form). The latent is captured by the RL
    pipeline subclass's ``prepare_latents`` override (which sees the upstream
    ``image_latents`` return value) and stamped as ``image_capture`` on
    ``DiffusionOutput.custom_output`` — the trainer-side replay's
    :meth:`QwenImageEditPlusDiffusionStep.predict_noise` packs it again
    before the token-concat, so the spatial form is what the conditions
    container expects (mirrors the trainsite
    :class:`QwenImageEditPlusVAEEncodeStage` output).
    """

    _MISSING_IMAGE_CAPTURE_MSG = (
        "build_response: Qwen-Image-Edit-Plus rollout returned no 'image_capture' on "
        "DiffusionOutput.custom_output. Check that RLQwenImageEditPlusPipeline's "
        "prepare_latents override ran in every DiT worker — the image_latents capture "
        "is required for trainer-side replay (predict_noise concatenates it onto the "
        "noise latent)."
    )

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """T2I text-capture conditions + Edit-Plus ``image_latent``.

        Calls ``super().build_conditions`` for the ``text`` / ``negative_text``
        slots (ragged-pad-concat of the per-request Qwen ``text_capture``
        dicts), then wraps the ``image_capture`` tensor as an
        :class:`ImageLatentCondition` and emits it under ``image_latent``.
        """
        cond_dict = super().build_conditions(req, per_request)
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        image_latents = self._collect_image_capture(diff_outputs)
        cond_dict["image_latent"] = ImageLatentCondition(latents=image_latents)
        return cond_dict

    def _collect_image_capture(self, diff_outputs: List[Any]) -> torch.Tensor:
        """Concatenate the per-request ``image_capture`` tensors along dim 0.

        Each request's ``image_capture`` is the spatial source-image latent
        ``[1, 16, H_i/8, W_i/8]`` (or ``[B_i, 16, ...]`` for multi-image-
        per-prompt, but the NFT recipe uses 1 image per prompt). The trainer-
        side replay expects a batched ``[B, 16, H/8, W/8]`` tensor; requests
        with differing image sizes would ragged-pad here, but the recipe pins
        a fixed generation size and the pre_process_func normalizes to
        ``vae_size`` so all source images share the same latent grid.
        """
        tensors: List[torch.Tensor] = []
        for d in diff_outputs:
            cap = (getattr(d, "custom_output", None) or {}).get("image_capture")
            if cap is None:
                raise RuntimeError(self._MISSING_IMAGE_CAPTURE_MSG)
            tensors.append(cap)
        # All source-image latents share the same vae_size-derived grid (the
        # pre_process_func normalizes), so dim-0 concat is safe. Guard anyway.
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) > 1:
            raise RuntimeError(
                f"build_response: Qwen-Image-Edit-Plus image_capture tensors have "
                f"heterogeneous shapes {sorted(shapes)} — expected a uniform grid "
                f"(the pre_process_func normalizes to vae_size). Check that all "
                f"source images in the batch have the same aspect ratio, or "
                f"extend the adapter to ragged-pad."
            )
        return torch.cat(tensors, dim=0)


@register_adapter("qwen_image_edit_plus_t2i")
class QwenImageEditPlusT2iAdapter(ModelAdapter):
    """Qwen-Image-Edit-Plus text+image → image (single diffusion stage, TP=1)."""

    stage_yaml = "qwen_image_edit_plus_t2i_rl.yaml"
    omni_mode = "text-to-image"
    # The Qwen2.5-VL tokenizer lives in the tokenizer/ subfolder; the worker
    # loads it and the single-stage path never calls build_prompt_tokens.
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = QwenImageEditPlusInputAdapter(self.modality, model_config=model_config)
        self.output_adapter = QwenImageEditPlusOutputAdapter(self.modality)

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is None:
            raise ValueError(f"modality={self.modality!r} requires req.primitives['image'] (Edit-Plus is edit-only).")

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        calls: List[GenerateCall] = []
        for idx in range(req.batch_size):
            # Upstream QwenImageEditPlusPipeline.forward consumes only req.prompts[0].
            single_req = req.slice(idx, idx + 1)
            call = self.input_adapter.build(single_req)[0]
            calls.append(
                GenerateCall(
                    prompts=call.prompts,
                    sampling=call.sampling,
                    group_by_request_id=False,
                )
            )
        return calls

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = [
    "QwenImageEditPlusInputAdapter",
    "QwenImageEditPlusOutputAdapter",
    "QwenImageEditPlusT2iAdapter",
]
