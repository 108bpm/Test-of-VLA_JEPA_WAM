"""Public encoder + action-conditioned latent world model interface."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor

from .predictor import build_predictor


@dataclass
class LatentWorldModelConfig:
    """Predictor configuration.

    Defaults match the VLA-JEPA setup for an 8-frame, two-view V-JEPA2 ViT-L
    encoder. ``latent_action_dim`` and ``num_action_tokens_per_timestep`` are
    intentionally independent of the encoder so callers can connect a policy,
    language model, or learned action tokenizer with their own representation.
    """

    num_video_frames: int = 8
    predictor_depth: int = 12
    predictor_num_heads: int = 8
    latent_action_dim: int = 2048
    num_action_tokens_per_timestep: int = 8
    num_views: int = 2


class LatentWorldModel(nn.Module):
    """Frozen V-JEPA2 encoder plus a trainable action-conditioned latent predictor.

    The model predicts one next latent state for every context state. Encode
    ``T`` raw frames; V-JEPA2 tubelets yield ``T / tubelet_size`` latent steps.
    For encoded states ``z_0, ..., z_N``, context is ``z_0, ..., z_{N-1}`` and
    targets are the time-aligned next states ``z_1, ..., z_N``. No assumption
    is made about how ``latent_actions`` are produced, enabling another
    project's policy, language model, or learned action encoder to supply them.
    """

    def __init__(self, encoder_path: Union[str, Path], config: Optional[LatentWorldModelConfig] = None):
        super().__init__()
        self.config = config or LatentWorldModelConfig()
        self.encoder = AutoModel.from_pretrained(str(encoder_path))
        self.video_processor = AutoVideoProcessor.from_pretrained(str(encoder_path))
        self.freeze_encoder()
        enc_cfg = self.encoder.config
        self.tubelet_size = enc_cfg.tubelet_size
        if self.config.num_video_frames % self.tubelet_size:
            raise ValueError("num_video_frames must be divisible by the encoder tubelet_size")
        self.latent_steps = self.config.num_video_frames // self.tubelet_size
        if self.latent_steps < 2:
            raise ValueError("num_video_frames must produce at least two V-JEPA latent steps")
        self.predictor = build_predictor(
            num_frames=self.latent_steps,
            img_size=(enc_cfg.image_size, enc_cfg.image_size),
            tubelet_size=1,
            depth=self.config.predictor_depth,
            num_heads=self.config.predictor_num_heads,
            embed_dim=enc_cfg.hidden_size * self.config.num_views,
            action_embed_dim=self.config.latent_action_dim,
            num_add_tokens=self.config.num_action_tokens_per_timestep,
        )

    def freeze_encoder(self):
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep the frozen teacher encoder in eval mode during predictor training."""
        super().train(mode)
        self.encoder.eval()
        return self

    @property
    def context_steps(self) -> int:
        return self.latent_steps - 1

    def preprocess_video(self, videos: torch.Tensor) -> torch.Tensor:
        """Preprocess uint8/raw videos `[B,V,T,C,H,W]` using the V-JEPA processor."""
        if videos.ndim != 6:
            raise ValueError("videos must have shape [B, V, T, C, H, W]")
        bsz, views, frames, channels, height, width = videos.shape
        if views != self.config.num_views or frames != self.config.num_video_frames:
            raise ValueError("video view/frame dimensions do not match LatentWorldModelConfig")
        flat = videos.reshape(bsz * views, frames, channels, height, width)
        processed = [self.video_processor(v, return_tensors="pt")["pixel_values_videos"] for v in flat]
        # Each processor call returns ``[1, T, C, H, W]``.  Restore both the
        # batch and view axes; dropping the leading batch axis here used to
        # make the public ``encode_video`` path fail its shape contract.
        processed_batch = torch.cat(processed, dim=0)
        return processed_batch.view(bsz, views, *processed_batch.shape[1:]).to(self.encoder.device)

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor, preprocessed: bool = False) -> torch.Tensor:
        """Encode multi-view video to `[B, latent_steps*patches, num_views*hidden]`.

        Set ``preprocessed=True`` when input already matches the V-JEPA
        processor's normalized `[B,V,T,C,H,W]` tensor format.
        """
        pixels = videos if preprocessed else self.preprocess_video(videos)
        if pixels.ndim != 6:
            raise ValueError("videos must have shape [B, V, T, C, H, W]")
        bsz, views = pixels.shape[:2]
        if views != self.config.num_views:
            raise ValueError("unexpected number of views")
        features = self.encoder.get_vision_features(pixel_values_videos=pixels.flatten(0, 1).to(self.encoder.device))
        # Preserve each sample's views together. This corrects the original
        # VLA-JEPA code's torch.chunk-by-view assumption for B > 1.
        return features.view(bsz, views, features.shape[1], features.shape[2]).permute(0, 2, 1, 3).flatten(2)

    def split_context_target(self, video_latents: torch.Tensor):
        """Return context and its time-aligned one-step-ahead target sequence.

        If encoded token blocks represent ``[z_0, z_1, ..., z_N]``, returns
        ``[z_0, ..., z_{N-1}]`` and ``[z_1, ..., z_N]``. This is the causal
        teacher-forcing objective used by the source VLA-JEPA predictor.
        """
        patches = video_latents.shape[1] // self.latent_steps
        if patches * self.latent_steps != video_latents.shape[1]:
            raise ValueError("encoder latent token count is incompatible with configured frame count")
        return video_latents[:, :-patches], video_latents[:, patches:]

    def load_predictor_checkpoint(self, checkpoint_path: Union[str, Path], strict: bool = True) -> Dict[str, list]:
        """Load only ``vj_predictor`` weights from a VLA-JEPA checkpoint.

        The full VLA-JEPA checkpoint also contains Qwen, V-JEPA encoder, and
        action-head weights.  Loading only this prefix keeps evaluation memory
        bounded and makes it explicit that no model component is retrained.
        ``mmap=True`` is used for local PyTorch archives so the unrelated
        multi-gigabyte tensors are not eagerly copied into RAM.
        """
        path = str(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        prefix = "vj_predictor."
        predictor_state = {
            key[len(prefix):]: value
            for key, value in checkpoint.items()
            if key.startswith(prefix)
        }
        if not predictor_state:
            raise KeyError(f"No {prefix!r} parameters found in {checkpoint_path}")
        result = self.predictor.load_state_dict(predictor_state, strict=strict)
        del checkpoint
        return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}

    def predict_from_latents(self, context_latents: torch.Tensor, latent_actions: torch.Tensor) -> torch.Tensor:
        """Predict a next latent block for each context step.

        Args:
            context_latents: `[B, context_steps*patches, num_views*encoder_hidden]`.
            latent_actions: `[B, context_steps*K, latent_action_dim]`, where K
                is `num_action_tokens_per_timestep`. Tokens for context step i
                should condition its transition to target step i + 1. Their
                source is intentionally external to this package.

        Returns:
            `[B, context_steps*patches, num_views*encoder_hidden]`, aligned
            with the next-state target sequence from :meth:`split_context_target`.
        """
        if context_latents.ndim != 3:
            raise ValueError("context_latents must have shape [B, context_steps * patches, D]")
        # Infer context length from the encoder patch grid first.  A single
        # block (256 tokens) is itself divisible by ``latent_steps`` (4), so
        # checking divisibility by the temporal length would incorrectly turn
        # C1 into C3 and demand 24 action tokens.
        patch_size = getattr(self.encoder.config, "patch_size", 16)
        patch_count = (self.encoder.config.image_size // patch_size) ** 2
        if context_latents.shape[1] % patch_count:
            raise ValueError("context_latents token count is not divisible by the patch grid")
        context_steps = context_latents.shape[1] // patch_count
        if not 1 <= context_steps <= self.context_steps:
            raise ValueError(f"context_latents must contain 1..{self.context_steps} latent steps")
        expected = context_steps * self.config.num_action_tokens_per_timestep
        if latent_actions.ndim != 3 or latent_actions.shape[0] != context_latents.shape[0]:
            raise ValueError("latent_actions batch dimension must match context_latents")
        if latent_actions.shape[1] != expected or latent_actions.shape[2] != self.config.latent_action_dim:
            raise ValueError(
                f"latent_actions must have shape [B, {expected}, {self.config.latent_action_dim}]"
            )
        # Dataset action tokens are stored as float16 to save space, whereas
        # the standalone predictor is normally loaded in float32.  Match the
        # predictor's parameter dtype/device at the boundary so callers do not
        # need an otherwise surprising manual cast.
        action_parameter = self.predictor.action_encoder.weight
        context_parameter = self.predictor.predictor_embed.weight
        context_latents = context_latents.to(device=context_parameter.device, dtype=context_parameter.dtype)
        latent_actions = latent_actions.to(device=action_parameter.device, dtype=action_parameter.dtype)
        return self.predictor(context_latents, latent_actions)

    def forward(self, videos: torch.Tensor, latent_actions: torch.Tensor, preprocessed: bool = False):
        """Return aligned `(predicted_next_latents, target_next_latents)` for training."""
        latents = self.encode_video(videos, preprocessed=preprocessed)
        context, target = self.split_context_target(latents)
        return self.predict_from_latents(context, latent_actions), target

    def loss(self, videos: torch.Tensor, latent_actions: torch.Tensor, preprocessed: bool = False) -> torch.Tensor:
        predicted, target = self(videos, latent_actions, preprocessed=preprocessed)
        return F.l1_loss(predicted, target)
