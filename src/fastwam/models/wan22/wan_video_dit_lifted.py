from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

from .wan_video_dit import WanVideoDiT

logger = get_logger(__name__)


class WanVideoDiTLiftedBranch(WanVideoDiT):
    """VideoJAM-style visual branch: own patch embedding + head, shared DiT trunk.

    The backbone ``WanVideoDiT`` (video expert) owns ``blocks``, time/text embeddings,
    and cross-attention paths. This branch only adds trainable input/output stems so
    track latents enter and exit through modality-specific linear layers while MoT
    mixed attention uses the shared transformer weights.
    """

    def __init__(
        self,
        backbone: WanVideoDiT,
        init_stems_from_backbone: bool = True,
    ):
        nn.Module.__init__(self)
        object.__setattr__(self, "_backbone", backbone)

        if init_stems_from_backbone:
            self.patch_embedding = deepcopy(backbone.patch_embedding)
            self.head = deepcopy(backbone.head)
        else:
            self.patch_embedding = deepcopy(backbone.patch_embedding)
            self.head = deepcopy(backbone.head)
            nn.init.zeros_(self.patch_embedding.weight)
            if self.patch_embedding.bias is not None:
                nn.init.zeros_(self.patch_embedding.bias)
            nn.init.zeros_(self.head.head.weight)
            if self.head.head.bias is not None:
                nn.init.zeros_(self.head.head.bias)

        self._mirror_backbone_config(backbone)

        stem_params = sum(p.numel() for p in self.patch_embedding.parameters()) + sum(
            p.numel() for p in self.head.parameters()
        )
        trunk_params = sum(p.numel() for p in backbone.blocks.parameters())
        logger.info(
            "WanVideoDiTLiftedBranch: stem params=%.2f M (shared trunk %.2f B)",
            stem_params / 1e6,
            trunk_params / 1e9,
        )

    def _mirror_backbone_config(self, backbone: WanVideoDiT) -> None:
        """Expose backbone modules/flags used by ``pre_dit`` / ``post_dit`` / MoT."""
        self.hidden_dim = backbone.hidden_dim
        self.in_dim = backbone.in_dim
        self.out_dim = backbone.out_dim
        self.freq_dim = backbone.freq_dim
        self.patch_size = backbone.patch_size
        self.num_heads = backbone.num_heads
        self.attn_head_dim = backbone.attn_head_dim
        self.seperated_timestep = backbone.seperated_timestep
        self.fuse_vae_embedding_in_latents = backbone.fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = backbone.video_attention_mask_mode
        self.action_conditioned = backbone.action_conditioned
        self.action_dim = backbone.action_dim
        self.action_group_causal_mask_mode = backbone.action_group_causal_mask_mode
        self.use_gradient_checkpointing = backbone.use_gradient_checkpointing
        self.has_image_pos_emb = backbone.has_image_pos_emb
        self.has_ref_conv = backbone.has_ref_conv
        self.control_adapter = backbone.control_adapter

        self.text_embedding = backbone.text_embedding
        self.time_embedding = backbone.time_embedding
        self.time_projection = backbone.time_projection
        self.freqs = backbone.freqs
        if getattr(backbone, "action_conditioned", False):
            self.action_embedding = backbone.action_embedding

    @property
    def blocks(self) -> nn.ModuleList:
        return self._backbone.blocks

    @classmethod
    def from_backbone(
        cls,
        backbone: WanVideoDiT,
        init_stems_from_backbone: bool = True,
    ) -> "WanVideoDiTLiftedBranch":
        return cls(backbone=backbone, init_stems_from_backbone=init_stems_from_backbone)

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self._backbone.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        return self._backbone.unpatchify(x, grid_size)
