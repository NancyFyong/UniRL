"""RL-aware Qwen-Image-Edit-Plus pipeline subclass.

Sibling of :class:`unirl.rollout.engine.vllm_omni.pipelines.qwen_image.pipeline.RLQwenImagePipeline`
with two Edit-Plus deltas driven by the upstream
``QwenImageEditPlusPipeline`` (see
``vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit_plus.py``):

- ``prepare_latents`` **takes** a leading ``images`` positional argument and
  **returns a tuple** ``(latents, image_latents)``. Slot layout:

      prepare_latents(images, batch, channels, h, w, dtype, device, gen, latents)
                     #0      #1      #2        #3 #4 #5     #6     #7   #8

  So ``latents`` is at index **8** (vs index 7 in the T2I ``QwenImagePipeline``).
  The ``images`` argument is the upstream-preprocessed ``vae_images`` list
  (populated by ``get_qwen_image_edit_plus_pre_process_func``); the driver
  never touches it. The injection override only rewrites the ``latents``
  slot (index 8) with the driver-authored x_T and re-dispatches to
  ``super().prepare_latents``, then captures both return values.

- ``image_latents`` (packed ``[B, S_img, C*4]``) is the VAE-encoded source
  image that the Edit-Plus ``diffuse`` loop concatenates onto the noise
  latent at every step. The trainer-side replay
  (:meth:`QwenImageEditPlusDiffusionStep.predict_noise`) needs the
  **spatial** ``[B, C, H_img, W_img]`` form (mirrors the trainside
  :class:`QwenImageEditPlusVAEEncodeStage` output), so the harvest unpacks
  the captured ``image_latents`` back to spatial and stamps it as
  ``image_capture`` on ``DiffusionOutput.custom_output``.

Trajectory shape
----------------
The SDE scheduler records the ``latents`` tensor passed to
``scheduler.step(pred, t, latents)``. In Edit-Plus's ``diffuse`` loop
(``cfg_parallel.py:125``) that tensor is the **noise-only** ``latents``
variable — the image-latent concat lives in a separate
``latent_model_input`` per step and is never written back to ``latents``.
So the recorded trajectory is noise-only ``[B, T+1, S_noise, C*4]``, exactly
like T2I. The harvest unpacks it to ``[B, T+1, C, H, W]`` unchanged from
the T2I pipeline.

Image-grid geometry
-------------------
The pre-process func writes ``vae_image_sizes`` (a list of
``(vae_width, vae_height)`` pixel pairs per source image) into
``prompt["additional_information"]``. After upstream's
``prepare_latents`` runs, the spatial image-latent grid is
``(vae_height // vae_scale_factor, vae_width // vae_scale_factor)``. The
harvest reads the first entry (NFT recipe is 1-image-per-prompt) to
unpack ``image_latents`` from ``[B, S_img, C*4]`` back to
``[B, C, H_img, W_img]``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    QwenImageEditPlusPipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    drain_trajectory_into,
    inject_latents,
    make_sde_scheduler,
    resolve_request_noise,
    stamp_custom_output,
)


class RLQwenImageEditPlusPipeline(QwenImageEditPlusPipeline):
    """Qwen-Image-Edit-Plus pipeline with the RL interception protocol installed.

    See :class:`RLQwenImagePipeline` for the protocol overview; this subclass
    only specializes the ``prepare_latents`` interception to the Edit-Plus
    tuple-return signature and adds the ``image_latents`` capture on harvest.
    """

    # Slot layout of upstream ``QwenImageEditPlusPipeline.prepare_latents``:
    #   (images, batch, channels, h, w, dtype, device, gen, latents)
    # ``images`` is index 0 (the upstream-preprocessed vae_images list);
    # ``latents`` is index 8 (vs index 7 in the T2I QwenImagePipeline).
    _LATENTS_IDX: int = 8
    _DTYPE_IDX: int = 5
    _DEVICE_IDX: int = 6

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # Upstream ``__init__`` constructs ``self.scheduler``; stash it as
        # the config donor for the SDE swap. We never swap back — our
        # scheduler is installed for the lifetime of this pipeline instance.
        self._upstream_scheduler: FlowMatchEulerDiscreteScheduler = self.scheduler
        # Conditioning-tap state: armed (reset to a fresh dict) every
        # request, filled by the tap's first/second call; the flag keeps the
        # install idempotent.
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        # Per-request x_T hand-off: armed every request, consumed once by the
        # ``prepare_latents`` override. ``None`` = upstream RNG fires.
        self._pending_initial_noise: Optional[torch.Tensor] = None
        # The request's normalized pixel H/W, stashed by ``forward`` for the
        # harvest-side trajectory unpack.
        self._harvest_hw: Optional[Tuple[int, int]] = None
        # Per-request source-image latent grid (H_img, W_img) in latent
        # space, stashed by ``forward`` from the request's
        # ``vae_image_sizes`` metadata for the harvest-side image_latents
        # unpack. ``None`` = no image (T2I degenerate; not used in the NFT
        # recipe but kept for parity with the step's degenerate path).
        self._harvest_image_hw: Optional[Tuple[int, int]] = None
        # Per-request capture of the upstream-computed ``image_latents``
        # (packed ``[B, S_img, C*4]``), filled by the ``prepare_latents``
        # override's return-value intercept. Consumed once by harvest.
        self._captured_image_latents: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # install — once per pipeline lifetime, idempotent
    # ------------------------------------------------------------------ #

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler (the from_config
        path keeps the dynamic-shift config keys ``prepare_timesteps`` reads
        for μ). Always installed, even for eta=0 flows (NFT) — per-request
        eta rides ``_arm_sde``."""
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        self.scheduler = make_sde_scheduler(self._upstream_scheduler.config)

    def _install_conditioning_tap(self) -> None:
        """Wrap ``encode_prompt`` to capture the text conditioning.

        Upstream Edit-Plus ``encode_prompt`` returns
        ``(prompt_embeds, prompt_embeds_mask)`` — same shape contract as T2I
        (the Qwen2.5-VL last hidden states after the chat-template prefix
        strip + matching mask; no pooled vector). Call routing per request:
        the first call fills the positive slot, the second (fired by upstream
        only under ``do_true_cfg``) the negative slot.
        """
        if self._conditioning_tap_installed:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            cap = pipeline_self._captured_conditioning
            if cap is not None:
                prompt_embeds, prompt_embeds_mask = result
                if "prompt_embeds" not in cap:
                    cap["prompt_embeds"] = detach_cpu(prompt_embeds)
                    cap["prompt_embeds_mask"] = detach_cpu(prompt_embeds_mask)
                elif "negative_prompt_embeds" not in cap:
                    cap["negative_prompt_embeds"] = detach_cpu(prompt_embeds)
                    cap["negative_prompt_embeds_mask"] = detach_cpu(prompt_embeds_mask)
            return result

        self.encode_prompt = tapped  # type: ignore[assignment]
        self._conditioning_tap_installed = True

    # ------------------------------------------------------------------ #
    # arm — every request (stale-leak guards)
    # ------------------------------------------------------------------ #

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate."""
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        self.scheduler.arm(eta=eta, sde_indices=extra.get("sde_indices"))

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T (batch slice or recipe row),
        still in the spatial ``[1, C, H, W]`` shape — packing happens at the
        injection point where upstream's grid geometry is in hand."""
        self._pending_initial_noise = resolve_request_noise(
            req, caller="RLQwenImageEditPlusPipeline._arm_initial_noise"
        )

    def _arm_conditioning_tap(self) -> None:
        """Fresh capture buffer so the tap records THIS request's encodes."""
        self._captured_conditioning = {}

    def _arm_image_capture(self, req: OmniDiffusionRequest) -> None:
        """Stash this request's source-image latent grid for the harvest
        unpack. Read from the pre-process func's ``vae_image_sizes`` metadata
        (a list of ``(vae_width, vae_height)`` pixel pairs); the latent grid
        is ``(vae_height // vae_scale_factor, vae_width // vae_scale_factor)``.

        Single-image-per-prompt is the NFT recipe's contract; multi-image
        would need a list-of-grids harvest, deferred (the V1 plan pins 1
        source image per prompt).
        """
        self._captured_image_latents = None
        prompts = getattr(req, "prompts", None) or []
        image_hw: Optional[Tuple[int, int]] = None
        for p in prompts:
            if isinstance(p, str):
                continue
            info = (p or {}).get("additional_information") or {}
            sizes = info.get("vae_image_sizes")
            if not sizes:
                continue
            if len(sizes) != 1:
                raise NotImplementedError(
                    "RLQwenImageEditPlusPipeline._arm_image_capture: multi-image-per-prompt "
                    f"is not supported (got {len(sizes)} source images). The V1 NFT recipe "
                    "pins 1 source image per prompt; multi-image needs a list-of-grids harvest."
                )
            vae_width, vae_height = sizes[0]
            image_hw = (
                int(vae_height) // int(self.vae_scale_factor),
                int(vae_width) // int(self.vae_scale_factor),
            )
            break
        self._harvest_image_hw = image_hw

    # ------------------------------------------------------------------ #
    # run-phase interception — upstream-called name, cannot be renamed
    # ------------------------------------------------------------------ #

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection + image_latents capture point.

        Upstream's ``latents is not None`` early-return skips the RNG draw
        but still packs the supplied latents, so the driver's spatial
        ``[1, C, H, W]`` noise is packed here first (the denoise loop runs
        in the transformer's ``[B, S, C*4]`` patch layout). The packed noise
        is written into the ``latents`` slot (index 8 of the positional
        call — Edit-Plus's leading ``images`` argument shifts the layout by
        one vs T2I).

        The return value is a tuple ``(latents, image_latents)``. Both are
        captured: ``latents`` flows into the denoise loop (and the scheduler
        records its trajectory), ``image_latents`` is stashed for the
        harvest-side ``image_capture`` stamp. Consume-once.
        """
        noise = self._pending_initial_noise
        if noise is not None:
            self._pending_initial_noise = None
            args, kwargs = inject_latents(
                args,
                kwargs,
                self._pack_pending_noise(noise, args),
                dtype_idx=self._DTYPE_IDX,
                device_idx=self._DEVICE_IDX,
                latents_idx=self._LATENTS_IDX,
            )
        result = super().prepare_latents(*args, **kwargs)
        # Edit-Plus returns ``(latents, image_latents)``; capture the
        # image_latents for the harvest-side stamp. ``image_latents`` may be
        # ``None`` when upstream received no ``images`` (T2I degenerate);
        # the NFT recipe always supplies vae_images via the pre-process func.
        if isinstance(result, tuple) and len(result) == 2:
            latents, image_latents = result
            if image_latents is not None:
                self._captured_image_latents = image_latents.detach().to("cpu")
            return latents, image_latents
        # Defensive: if upstream ever changes the return shape, fail loud
        # rather than silently dropping the image_latents capture.
        raise RuntimeError(
            "RLQwenImageEditPlusPipeline.prepare_latents: upstream returned "
            f"{type(result).__name__} (expected a 2-tuple `(latents, image_latents)` "
            "from QwenImageEditPlusPipeline.prepare_latents). Upstream may have "
            "changed its return signature — update this override."
        )

    def _pack_pending_noise(self, noise: torch.Tensor, args: tuple) -> torch.Tensor:
        """Spatial ``[B, C, h, w]`` x_T → packed ``[B, S, C*4]``, validated
        against the call site's grid geometry. Upstream calls Edit-Plus's
        ``prepare_latents(images, batch_size, num_channels_latents, height,
        width, dtype, device, generator, latents)`` with all args positional
        and pixel-space H/W at indices 3/4 (vs 2/3 in T2I — the leading
        ``images`` argument shifts the layout). The latent grid is
        ``2 * (px // (vae_sf * 2))`` per side (the divisible-by-2 packing
        constraint).
        """
        # Slot layout: images@0, batch@1, channels@2, height@3, width@4, ...
        if len(args) < 5:
            raise RuntimeError(
                "RLQwenImageEditPlusPipeline._pack_pending_noise: expected upstream's "
                f"fully positional prepare_latents call; got {len(args)} positional args."
            )
        batch, channels = int(args[1]), int(args[2])
        grid_h = 2 * (int(args[3]) // (self.vae_scale_factor * 2))
        grid_w = 2 * (int(args[4]) // (self.vae_scale_factor * 2))
        if tuple(noise.shape) != (batch, channels, grid_h, grid_w):
            raise RuntimeError(
                "RLQwenImageEditPlusPipeline: driver x_T shape "
                f"{tuple(noise.shape)} does not match the worker latent grid "
                f"[{batch}, {channels}, {grid_h}, {grid_w}] for "
                f"{int(args[3])}x{int(args[4])} px — check the recipe's "
                "init_noise_latent_shape / initial_noise_batch."
            )
        return self._pack_latents(noise, batch, channels, grid_h, grid_w)

    # ------------------------------------------------------------------ #
    # harvest — export onto the wire
    # ------------------------------------------------------------------ #

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        """Drain the SDE scheduler's noise-only trajectory and unpack to
        spatial ``[B, T+1, C, H, W]``. The Edit-Plus ``diffuse`` loop never
        writes the image-latent concat back into the per-step ``latents``
        variable (the concat lives in a separate ``latent_model_input``
        per step), so the recorded trajectory is noise-only — the unpack
        is identical to T2I."""
        if not isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        drain_trajectory_into(out, self.scheduler)
        if out.trajectory_latents is not None:
            out.trajectory_latents = self._unpack_trajectory(out.trajectory_latents)

    def _unpack_trajectory(self, packed: torch.Tensor) -> torch.Tensor:
        """Packed ``[B, T+1, S, C*4]`` trajectory → spatial
        ``[B, T+1, C, H, W]`` (the trainer's ``LatentSegment`` shape).

        Upstream's ``_unpack_latents`` takes pixel H/W and returns the 5D
        video-VAE shape ``[N, C, 1, h, w]``; the singleton frame dim is
        squeezed out to match the trainside spatial convention
        (``models/qwen_image/diffusion.py`` keeps ``[B, K, C, H, W]``).
        """
        if self._harvest_hw is None:
            raise RuntimeError(
                "RLQwenImageEditPlusPipeline._unpack_trajectory: no stashed H/W — forward() did not run before harvest."
            )
        height, width = self._harvest_hw
        b, t1 = packed.shape[0], packed.shape[1]
        flat = self._unpack_latents(packed.reshape(b * t1, *packed.shape[2:]), height, width, self.vae_scale_factor)
        flat = flat.squeeze(2)  # [B*(T+1), C, 1, h, w] → [B*(T+1), C, h, w]
        return flat.reshape(b, t1, *flat.shape[1:])

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        """Stamp the text capture (positive + optional negative) and the
        image_latents capture onto ``custom_output`` for the response adapter."""
        if self._captured_conditioning:
            stamp_custom_output(out, "text_capture", self._captured_conditioning)
        if self._captured_image_latents is not None:
            stamp_custom_output(out, "image_capture", self._unpack_image_latents(self._captured_image_latents))

    def _unpack_image_latents(self, image_latents: torch.Tensor) -> torch.Tensor:
        """Packed ``[B, S_img, C*4]`` source-image latent → spatial
        ``[B, C, H_img, W_img]`` (the trainer-side replay's
        :class:`ImageLatentCondition` contract). The grid is read from the
        per-request ``vae_image_sizes`` metadata stashed in
        :meth:`_arm_image_capture`.

        Upstream's ``_unpack_latents`` takes pixel H/W; the latent grid
        passed here is already in latent space (``H_img, W_img``), so the
        pixel H/W to pass upstream is ``H_img * vae_scale_factor`` (which
        upstream then divides back by ``vae_scale_factor`` internally —
        the function's only use of the scale factor is the latent-grid
        recompute, so this round-trips cleanly).
        """
        if self._harvest_image_hw is None:
            raise RuntimeError(
                "RLQwenImageEditPlusPipeline._unpack_image_latents: no stashed "
                "image H/W — _arm_image_capture did not find vae_image_sizes in "
                "the request's additional_information. Check that the pre-process "
                "func ran (it should populate vae_image_sizes from the source PIL)."
            )
        h_lat, w_lat = self._harvest_image_hw
        # Upstream's _unpack_latents expects pixel H/W; pass latent grid × vae_sf.
        h_px = h_lat * int(self.vae_scale_factor)
        w_px = w_lat * int(self.vae_scale_factor)
        spatial = self._unpack_latents(image_latents, h_px, w_px, self.vae_scale_factor)
        # Squeeze the singleton frame dim: [B, C, 1, H_img, W_img] → [B, C, H_img, W_img]
        return spatial.squeeze(2)

    # ------------------------------------------------------------------ #
    # the protocol
    # ------------------------------------------------------------------ #

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        self._install_sde_scheduler()
        self._install_conditioning_tap()

        self._arm_sde(req)
        self._arm_initial_noise(req)
        self._arm_conditioning_tap()
        self._arm_image_capture(req)
        # Mirror upstream forward's H/W resolution (defaults + 16-alignment)
        # so the harvest unpack uses the exact grid the loop ran on. The
        # Edit-Plus pre-process func may have already set height/width on
        # req.sampling_params from the source image's aspect ratio; honor
        # that (the NFT recipe pins 384² so this is a no-op there, but the
        # dynamic-sizing path needs to track whatever upstream chose).
        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        self._harvest_hw = (int(height), int(width))

        # Delegate the entire denoise pipeline (prompt encoding with the
        # source-image condition_images, latent prep, timestep build, the
        # Edit-Plus diffusion loop with image_latent token-concat, VAE
        # decode) to upstream; the installed tap/injector fire inside.
        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        return out


__all__ = ["RLQwenImageEditPlusPipeline"]
