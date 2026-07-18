"""Action-conditioned predictor copied from VLA-JEPA with no VLA dependency."""

import math
from functools import partial

import torch
import torch.nn as nn

from .vj2_modules import ACBlock as Block
from .vj2_modules import build_action_block_causal_attention_mask
from .vj2_tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    """Predict one next V-JEPA patch-token state per context time step.

    ``actions`` is deliberately generic: it is a sequence of external latent
    action vectors.  For T context frames, pass ``[B, T * K, action_dim]``;
    K is ``num_action_tokens_per_timestep``. The output has the same token
    shape as the context and is aligned to the next state of each step by the
    :class:`latent_world_model.LatentWorldModel` wrapper.
    """

    def __init__(
        self, img_size=(256, 256), patch_size=16, num_frames=4, tubelet_size=1,
        embed_dim=2048, predictor_embed_dim=1024, depth=12, num_heads=8,
        mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
        attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm,
        init_std=0.02, uniform_power=True, use_silu=False, wide_silu=True,
        is_frame_causal=True, use_activation_checkpointing=False, use_rope=True,
        action_embed_dim=2048, use_extrinsics=False, num_add_tokens=8,
    ):
        super().__init__()
        self.use_extrinsics = use_extrinsics
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_height = self.img_height // patch_size
        self.grid_width = self.img_width // patch_size
        self.use_activation_checkpointing = use_activation_checkpointing
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            Block(use_rope=use_rope, grid_size=self.grid_height,
                  dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  act_layer=nn.SiLU if use_silu else nn.GELU, wide_silu=wide_silu,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()
        self.attn_mask = build_action_block_causal_attention_mask(
            num_frames // tubelet_size, self.grid_height, self.grid_width, add_tokens=num_add_tokens
        ) if is_frame_causal else None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1)

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(self, context_latents, actions, extrinsics=None):
        """Return predicted patch latents with shape equal to ``context_latents``."""
        x = self.predictor_embed(context_latents)
        batch, context_tokens, dim = x.shape
        tokens_per_frame = self.grid_height * self.grid_width
        if context_tokens % tokens_per_frame:
            raise ValueError("context_latents token count must be a multiple of H_patches * W_patches")
        steps = context_tokens // tokens_per_frame
        if actions.ndim != 3 or actions.shape[0] != batch or actions.shape[1] % steps:
            raise ValueError("latent_actions must have shape [B, context_steps * K, action_dim]")
        action_tokens = self.action_encoder(actions).view(batch, steps, -1, dim)
        cond_count = action_tokens.shape[2]
        x = x.view(batch, steps, tokens_per_frame, dim)
        if self.use_extrinsics:
            if extrinsics is None:
                raise ValueError("extrinsics is required when use_extrinsics=True")
            extrinsics = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            cond_count += 1
            x = torch.cat([action_tokens, extrinsics, x], dim=2).flatten(1, 2)
        else:
            x = torch.cat([action_tokens, x], dim=2).flatten(1, 2)
        mask = self.attn_mask[:x.size(1), :x.size(1)].to(x.device) if self.attn_mask is not None else None
        for block in self.predictor_blocks:
            x = block(x, mask=None, attn_mask=mask, T=steps, H=self.grid_height,
                      W=self.grid_width, action_tokens=cond_count)
        x = x.view(batch, steps, cond_count + tokens_per_frame, dim)[:, :, cond_count:].flatten(1, 2)
        return self.predictor_proj(self.predictor_norm(x))


def build_predictor(**kwargs):
    return VisionTransformerPredictorAC(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
