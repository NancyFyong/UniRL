"""QwenImageEditPlus VAE stages — source-image encoder + (reused) decoder.

``QwenImageEditPlusVAEEncodeStage`` is the one genuinely new stage: it
turns a source ``Images`` primitive into an :class:`ImageLatentCondition`
carrying the VAE-encoded spatial latent ``[B, 16, H/8, W/8]``. The diffusion
step (:class:`QwenImageEditPlusDiffusionStep`) packs both the noise latent
and this image latent with the same 2×2 channel-pack before concatenating
along the token dimension — mirrors ``vde_editplus.py:232`` and the
FLUX.2-Klein image-edit pattern (``flux2_klein/vae.py:86-158``).

The decode side reuses :class:`unirl.models.qwen_image.QwenImageVAEDecodeStage`
unchanged (same VAE, same 5D un-normalization math) — re-exported here so
the Edit-Plus package is self-contained.
"""

from __future__ import annotations

import torch

from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import ImageLatentCondition
from unirl.types.primitives import Images

from .bundle import QwenImageEditPlusBundle


class QwenImageEditPlusVAEEncodeStage(EncodeStage[Images, ImageLatentCondition]):
    """Encode a source image into a VAE-latent condition for token concat.

    Pipeline:

    1. Resize source pixels to the generation ``(height, width)``. The data
       source loads condition images at native resolution (arbitrary H×W),
       but the Qwen-Image VAE requires H, W divisible by 16 (8× VAE +
       2× patch), and a consistent token count across a GRPO group needs
       a fixed size. Using the generation size satisfies both (recipe
       sizes are multiples of 16) and matches the edited-image resolution.
       Mirrors ``flux2_klein/vae.py:125-135``.
    2. ``[0, 1] → [-1, 1]`` (VAE input convention).
    3. ``vae.encode(x).latent_dist.mode()`` — deterministic (matches
       diffusers' ``retrieve_latents(sample_mode="argmax")`` so
       rollout/replay don't drift). Mirrors ``flux2_klein/vae.py:141``.
    4. Return the spatial latent ``[B, 16, H/8, W/8]`` wrapped in an
       :class:`ImageLatentCondition`. **Do NOT** ``_pack_latents`` here —
       the diffusion step packs both noise and image latents together so
       they share the same 2×2 pack logic; the condition carries the
       spatial latent (mirrors ``Flux2KleinConditions.image_latent``).
    """

    def __init__(self, bundle: QwenImageEditPlusBundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, images: Images, *, height: int, width: int) -> ImageLatentCondition:
        """Encode source pixels into an :class:`ImageLatentCondition`.

        Args:
            images: source ``Images`` with ``pixels`` ``[B, 3, H, W]`` in
                ``[0, 1]``.
            height, width: generation grid (must be divisible by 16: 8× VAE
                downsample + 2× patchify).

        Returns:
            :class:`ImageLatentCondition` with ``latents`` shape
            ``[B, 16, H/8, W/8]`` on the bundle device, in the bundle dtype.
        """
        pixels = images.pixels
        if pixels is None or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"QwenImageEditPlusVAEEncodeStage.encode: expected pixels "
                f"[B, 3, H, W] in [0,1], got shape "
                f"{None if pixels is None else tuple(pixels.shape)}"
            )
        if int(height) % 16 != 0 or int(width) % 16 != 0:
            raise ValueError(
                f"QwenImageEditPlusVAEEncodeStage.encode: height ({height}) and "
                f"width ({width}) must be divisible by 16 (8× VAE + 2× patchify)"
            )

        vae = self.bundle.vae
        device = self.bundle.device
        dtype = self.bundle.dtype
        vae_f32 = vae.to(torch.float32)

        pixels = pixels.to(device=device, dtype=torch.float32)
        if int(pixels.shape[-2]) != int(height) or int(pixels.shape[-1]) != int(width):
            pixels = torch.nn.functional.interpolate(
                pixels, size=(int(height), int(width)), mode="bilinear", align_corners=False
            )

        # [0, 1] → [-1, 1] (VAE input convention).
        scaled = pixels * 2.0 - 1.0

        # Deterministic latents (mode). AutoencoderKLQwenImage.encode returns
        # a latent_dist; .mode() is the deterministic posterior mean.
        image_latents = vae_f32.encode(scaled).latent_dist.mode()  # [B, 16, H/8, W/8]

        return ImageLatentCondition(latents=image_latents.to(dtype=dtype))


__all__ = ["QwenImageEditPlusVAEEncodeStage"]
