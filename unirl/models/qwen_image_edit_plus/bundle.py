"""QwenImageEditPlusBundle — thin subclass of :class:`QwenImageBundle`.

The Edit-Plus checkpoint ships the same weight layout as base Qwen-Image
(``transformer/`` + ``vae/`` + ``text_encoder/`` + ``tokenizer/`` +
``scheduler/`` subfolders); only ``transformer/config.json`` differs
(``in_channels=64`` to absorb the source-image latent concat). The base
:meth:`QwenImageBundle.from_config` / ``_from_config_locked`` / meta-init
path / ``fcntl`` serialization all apply unchanged — ``in_channels`` is
read from the checkpoint automatically. This subclass exists so the
Edit-Plus package is self-contained under ``unirl/models/<model_name>/``
per the add-model-bundle skill, and so recipes can wire
``_target_: ...QwenImageEditPlusBundle.from_config``.
"""

from __future__ import annotations

from unirl.models.qwen_image.bundle import QwenImageBundle

from .config import QwenImageEditPlusPipelineConfig


class QwenImageEditPlusBundle(QwenImageBundle):
    """Qwen-Image-Edit-Plus bundle: transformer (in_channels=64) + VAE +
    Qwen-VL text encoder + scheduler.

    Inherits :meth:`from_config` / :meth:`_from_config_locked` from
    :class:`QwenImageBundle` unchanged. The config type annotation widens
    to :class:`QwenImageEditPlusPipelineConfig` for documentation; the
    runtime fields are identical, so the inherited ``from_config`` body
    reads them without modification.
    """

    @classmethod
    def from_config(cls, config: QwenImageEditPlusPipelineConfig) -> "QwenImageEditPlusBundle":
        """Load all Edit-Plus components from a HuggingFace-layout checkpoint.

        Delegates to :meth:`QwenImageBundle.from_config` — the Edit-Plus
        checkpoint is structurally identical to base Qwen-Image; only the
        transformer's input projection width differs, and that is read
        from ``transformer/config.json`` by diffusers automatically.
        """
        # ``QwenImageBundle.from_config`` reads only attributes that exist
        # on ``QwenImageEditPlusPipelineConfig`` (field-for-field compatible),
        # so the upcast is safe.
        return super().from_config(config)  # type: ignore[return-value]


__all__ = ["QwenImageEditPlusBundle"]
