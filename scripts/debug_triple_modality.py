"""Trace data shapes through the TRIPLE-modality FastWAM model.

A debug entrypoint for understanding how ``FastWAMTriple`` (video + point-track
+ action) routes data through the ``MoT`` (Mixture-of-Transformers) trunk:

  - Modality-specific stems  (``WanVideoDiT.pre_dit``, ``WanVideoDiTLiftedBranch.pre_dit``,
    ``ActionDiT.pre_dit``)
  - SHARED mixed self-attention inside ``MoT.forward`` (the only place
    cross-modality information flows)
  - Modality-specific heads  (``...post_dit``)

Two modes:

  1. **Default (VAE-free)** — construct tiny experts and drive them with
     synthetic latents whose shapes match what ``FastWAMTriple.training_loss``
     would see *after* VAE encoding. Runs in ~5s on CPU, no weights downloaded.

  2. **``--with-vae``** — additionally load the *real* Wan2.2 VAE from
     ``./checkpoints/`` (or ``$DIFFSYNTH_MODEL_BASE_PATH``), encode a synthetic
     RGB clip, and feed the resulting latents through the tiny experts. Shows
     the pixel → latent shape contraction (and a decode at the end). The DiT
     and T5 text encoder are NOT loaded — only the VAE.

Usage::

    # Fast, deps-free
    FASTWAM_DEBUG_SHAPES=1 python scripts/debug_triple_modality.py

    # Adds pixel-space pipeline through the real VAE
    FASTWAM_DEBUG_SHAPES=1 python scripts/debug_triple_modality.py --with-vae

    # Or run under one of the VS Code "Debug TRIPLE modality ..." launch
    # configs, which set FASTWAM_DEBUG_SHAPES=1 automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

# Default to printing model-internal shape traces unless the user explicitly
# disabled them; must happen before importing fastwam.* so the import-time
# env-var check picks it up.
os.environ.setdefault("FASTWAM_DEBUG_SHAPES", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from fastwam.models.wan22.action_dit import ActionDiT  # noqa: E402
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs  # noqa: E402
from fastwam.models.wan22.mot import MoT  # noqa: E402
from fastwam.models.wan22.wan_video_dit import WanVideoDiT  # noqa: E402
from fastwam.models.wan22.wan_video_dit_lifted import WanVideoDiTLiftedBranch  # noqa: E402
from fastwam.utils.shape_debug import dprint, dsection  # noqa: E402


def _human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    # Keep shapes tiny so the script runs in ~seconds on CPU and is easy to step through.
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num-layers", type=int, default=2, help="DiT layers per expert.")
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=4)
    # attn_head_dim must be a multiple of 6 so the 3D RoPE split
    # `dim - 2*(dim//3)` and `dim//3` are both even (see precompute_freqs_cis_3d).
    p.add_argument("--attn-head-dim", type=int, default=48)
    # Latent video shape [B, in_dim=48, T, H, W]: 48 is Wan22 VAE z_dim and
    # is fixed because the patch_embedding Conv3d uses it.
    p.add_argument("--latent-t", type=int, default=3)
    p.add_argument("--latent-h", type=int, default=8)
    p.add_argument("--latent-w", type=int, default=8)
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--context-len", type=int, default=8)
    p.add_argument("--text-dim", type=int, default=4096)
    # VAE mode (loads the real Wan2.2 VAE; --latent-* ignored)
    p.add_argument(
        "--with-vae",
        action="store_true",
        help="Load real Wan2.2 VAE and encode a synthetic RGB clip into latents.",
    )
    # Pixel-space dimensions used in --with-vae mode (must satisfy VAE constraints:
    # video_frames % temporal_downsample_factor == 1; H,W multiples of upsampling_factor=16).
    p.add_argument("--video-frames", type=int, default=5)
    p.add_argument("--video-height", type=int, default=64)
    p.add_argument("--video-width", type=int, default=64)
    p.add_argument(
        "--vae-model-id",
        default="Wan-AI/Wan2.2-TI2V-5B",
        help="HF/ModelScope model id used to locate Wan2.2_VAE under DIFFSYNTH_MODEL_BASE_PATH.",
    )
    return p.parse_args()


def build_experts(args: argparse.Namespace, dtype: torch.dtype):
    """Build (video, track, action) experts with tiny shared-shape configs."""
    video_dit_kwargs = dict(
        hidden_dim=args.hidden_dim,
        in_dim=48,
        ffn_dim=args.ffn_dim,
        out_dim=48,
        text_dim=args.text_dim,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=args.num_heads,
        attn_head_dim=args.attn_head_dim,
        num_layers=args.num_layers,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False,
        action_dim=args.action_dim,
        use_gradient_checkpointing=False,
    )
    video_expert = WanVideoDiT(**video_dit_kwargs).to(device=args.device, dtype=dtype)

    # Track expert reuses the *video* DiT blocks; only patch_embedding + head are new.
    track_expert = WanVideoDiTLiftedBranch.from_backbone(
        backbone=video_expert,
        init_stems_from_backbone=True,
    ).to(device=args.device, dtype=dtype)

    action_expert = ActionDiT(
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        ffn_dim=args.ffn_dim,
        text_dim=args.text_dim,
        freq_dim=256,
        eps=1e-6,
        num_heads=args.num_heads,
        attn_head_dim=args.attn_head_dim,
        num_layers=args.num_layers,
        use_gradient_checkpointing=False,
    ).to(device=args.device, dtype=dtype)

    return video_expert, track_expert, action_expert


def summarize_layer_organization(
    video_expert: WanVideoDiT,
    track_expert: WanVideoDiTLiftedBranch,
    action_expert: ActionDiT,
    mot: MoT,
) -> None:
    """Print which parameters are modality-specific vs shared in MoT."""
    print()
    print("=" * 80)
    print("TRIPLE-modality layer organization")
    print("=" * 80)

    def _params(mod: torch.nn.Module) -> int:
        return sum(p.numel() for p in mod.parameters())

    # video_expert.blocks IS track_expert.blocks (lifted branch).
    blocks_shared_video_track = video_expert.blocks is track_expert.blocks
    text_shared = video_expert.text_embedding is track_expert.text_embedding
    time_shared = video_expert.time_embedding is track_expert.time_embedding

    rows = [
        ("video.patch_embedding (Conv3d)", "PER-MODALITY", _params(video_expert.patch_embedding)),
        ("video.text_embedding (MLP)",    "PER-MODALITY", _params(video_expert.text_embedding)),
        ("video.time_embedding (MLP)",    "PER-MODALITY", _params(video_expert.time_embedding)),
        ("video.time_projection (Linear)","PER-MODALITY", _params(video_expert.time_projection)),
        ("video.blocks (DiT trunk)",      "PER-MODALITY*", _params(video_expert.blocks)),
        ("video.head",                    "PER-MODALITY", _params(video_expert.head)),
        ("",                              "",              0),
        ("track.patch_embedding (Conv3d)","PER-MODALITY (own)", _params(track_expert.patch_embedding)),
        ("track.head",                    "PER-MODALITY (own)", _params(track_expert.head)),
        ("track.text_embedding",          f"SHARED w/ video ({text_shared})", 0),
        ("track.time_embedding",          f"SHARED w/ video ({time_shared})", 0),
        ("track.blocks",                  f"SHARED w/ video ({blocks_shared_video_track})", 0),
        ("",                              "",              0),
        ("action.action_encoder (Linear)","PER-MODALITY", _params(action_expert.action_encoder)),
        ("action.text_embedding (MLP)",   "PER-MODALITY", _params(action_expert.text_embedding)),
        ("action.time_embedding (MLP)",   "PER-MODALITY", _params(action_expert.time_embedding)),
        ("action.time_projection",        "PER-MODALITY", _params(action_expert.time_projection)),
        ("action.blocks (DiT trunk)",     "PER-MODALITY", _params(action_expert.blocks)),
        ("action.head (Linear)",          "PER-MODALITY", _params(action_expert.head)),
    ]
    for name, kind, n in rows:
        if not name:
            print("-" * 80)
            continue
        n_str = _human(n) if n else "(weights aliased to backbone)"
        print(f"  {name:36s} {kind:32s} {n_str}")
    print("-" * 80)
    print(
        "  * video.blocks holds the actual transformer weights; track.blocks is just a\n"
        "    @property returning video.blocks. So video & track SHARE the DiT trunk."
    )
    print()
    print("Per-block components (each DiTBlock has these, replicated per expert,")
    print("but track shares video's block list, so its block params == video's):")
    print("  - modulation (1,6,D)        : per-block AdaLN params")
    print("  - norm1, norm2, norm3       : LayerNorms (norm3 has affine, others not)")
    print("  - self_attn: q, k, v, o     : Q/K/V/output Linears")
    print("  - self_attn: norm_q, norm_k : RMSNorms before RoPE")
    print("  - cross_attn (q, k, v, o)   : attends to (text) context")
    print("  - ffn (Linear-GELU-Linear)  : MLP")
    print()
    print("MoT inside one layer (per training step):")
    print("  for each expert: build modality-specific Q/K/V (via own block.self_attn.{q,k,v})")
    print("  concat -> q_cat, k_cat, v_cat across [video | track | action]")
    print("  SHARED: F.scaled_dot_product_attention(q_cat, k_cat, v_cat, attn_mask)")
    print("           ^^^ this is the only place cross-modality information mixes ^^^")
    print("  split mixed output by expert sequence length")
    print("  per expert: self_attn.o -> + residual -> cross_attn(text) -> FFN")
    print()
    print(f"MoT.num_layers={mot.num_layers}  expert_order={mot.expert_order}")
    print("=" * 80)
    print()


def load_real_vae(args: argparse.Namespace, dtype: torch.dtype, device: str):
    """Load *only* the Wan2.2 VAE (skip DiT + T5) using the project loader helpers.

    Reads from ``./checkpoints/`` by default; override with the env var
    ``DIFFSYNTH_MODEL_BASE_PATH``. If files are missing the loader will attempt
    a network download unless ``DIFFSYNTH_SKIP_DOWNLOAD=true``.
    """
    dsection(f"Loading real VAE from model_id={args.vae_model_id}")
    _, _, vae_config, _ = _resolve_configs(
        model_id=args.vae_model_id,
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=dtype,
        device=device,
    )
    dprint(
        "VAE.metadata",
        z_dim=int(vae.model.z_dim),
        temporal_downsample_factor=int(vae.temporal_downsample_factor),
        upsampling_factor=int(vae.upsampling_factor),
        path=str(vae_config.path),
    )
    return vae


def encode_pixels_with_vae(
    args: argparse.Namespace,
    vae,
    dtype: torch.dtype,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Build a synthetic video tensor in pixel space and VAE-encode it to latents.

    Returns the same keys as ``make_synthetic_inputs`` so the downstream code
    path is identical. Action / context / timesteps are still random — only the
    video and track latents come from the real VAE.
    """
    B = args.batch
    T_pix, H_pix, W_pix = args.video_frames, args.video_height, args.video_width

    upsample = int(vae.upsampling_factor)
    temporal_factor = int(vae.temporal_downsample_factor)
    if H_pix % upsample != 0 or W_pix % upsample != 0:
        raise ValueError(
            f"--video-height ({H_pix}) and --video-width ({W_pix}) must be multiples of "
            f"vae.upsampling_factor={upsample}."
        )
    if (T_pix - 1) % temporal_factor != 0:
        raise ValueError(
            f"--video-frames ({T_pix}) must satisfy (T-1) % {temporal_factor} == 0 "
            f"(temporal_downsample_factor)."
        )

    # Pixels in [-1, 1] (the convention FastWAM uses; see runtime.run_inference).
    video_pixels = torch.rand(B, 3, T_pix, H_pix, W_pix, dtype=dtype, device=device) * 2.0 - 1.0
    track_pixels = torch.rand_like(video_pixels)

    dsection("VAE encode: pixel-space -> latent-space")
    dprint("VAE.input.video_pixels", v=video_pixels)
    dprint("VAE.input.track_pixels", v=track_pixels)

    video_latents = vae.encode(video_pixels, device=device)
    track_latents = vae.encode(track_pixels, device=device)

    dprint(
        "VAE.output.latents",
        video_latents=video_latents,
        track_latents=track_latents,
        contraction=f"[B,3,{T_pix},{H_pix},{W_pix}] -> "
        f"[B,{int(vae.model.z_dim)},{(T_pix - 1) // temporal_factor + 1},"
        f"{H_pix // upsample},{W_pix // upsample}]",
    )

    # Reuse the synthetic-inputs builder for context / timesteps / action so we
    # don't repeat the conditioning-tensor code path.
    rest = _build_conditioning_only(args, dtype=dtype, device=device)
    return {"video_latents": video_latents, "track_latents": track_latents, **rest}


def _build_conditioning_only(
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: str,
) -> Dict[str, torch.Tensor]:
    B = args.batch
    return dict(
        action=torch.randn(B, args.action_horizon, args.action_dim, dtype=dtype, device=device),
        context=torch.randn(B, args.context_len, args.text_dim, dtype=dtype, device=device),
        context_mask=torch.ones(B, args.context_len, dtype=torch.bool, device=device),
        timestep_video=torch.rand(B, dtype=dtype, device=device),
        timestep_track=torch.rand(B, dtype=dtype, device=device),
        timestep_action=torch.rand(B, dtype=dtype, device=device),
    )


def make_synthetic_inputs(
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: str,
    z_dim: int = 48,
) -> Dict[str, torch.Tensor]:
    """Synthesize the tensors that ``FastWAMTriple.build_inputs`` would emit
    after VAE encoding + text-encoder caching."""
    B = args.batch
    video_latents = torch.randn(B, z_dim, args.latent_t, args.latent_h, args.latent_w, dtype=dtype, device=device)
    track_latents = torch.randn_like(video_latents)
    return {
        "video_latents": video_latents,
        "track_latents": track_latents,
        **_build_conditioning_only(args, dtype=dtype, device=device),
    }


def build_attention_mask(
    video_seq_len: int,
    track_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    video_expert: WanVideoDiT,
    track_expert: WanVideoDiTLiftedBranch,
    device: str,
) -> torch.Tensor:
    """Rebuilds the same mask FastWAMTriple._build_mot_attention_mask creates."""
    total = video_seq_len + track_seq_len + action_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)

    v0, v1 = 0, video_seq_len
    t0, t1 = video_seq_len, video_seq_len + track_seq_len
    a0, a1 = video_seq_len + track_seq_len, total

    mask[v0:v1, v0:v1] = video_expert.build_video_to_video_mask(
        video_seq_len=video_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        device=torch.device(device),
    )
    mask[t0:t1, t0:t1] = track_expert.build_video_to_video_mask(
        video_seq_len=track_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        device=torch.device(device),
    )
    mask[t0:t1, v0:v1] = True
    mask[a0:a1, a0:a1] = True
    first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
    mask[a0:a1, v0 : v0 + first_frame_tokens] = True
    return mask


def main() -> None:
    """fastwam_triple.training_loss (line 324)"""
    args = parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # ----- (optional) Real VAE -----
    vae = None
    z_dim = 48
    if args.with_vae:
        vae = load_real_vae(args, dtype=dtype, device=args.device)
        z_dim = int(vae.model.z_dim)

    dsection("Build tiny TRIPLE model (video + track + action)")
    video_expert, track_expert, action_expert = build_experts(args, dtype=dtype)
    if z_dim != video_expert.in_dim:
        raise ValueError(
            f"VAE z_dim={z_dim} does not match video expert in_dim={video_expert.in_dim}. "
            "The video DiT's patch_embedding expects exactly z_dim input channels."
        )
    mot = MoT(
        mixtures={"video": video_expert, "track": track_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    ).to(device=args.device, dtype=dtype)
    summarize_layer_organization(video_expert, track_expert, action_expert, mot)

    if vae is not None:
        dsection("Stage 0: VAE encode pixel-space inputs")
        inputs = encode_pixels_with_vae(args, vae, dtype=dtype, device=args.device)
    else:
        dsection("Synthesize inputs (replaces VAE encode + T5 text encoder)")
        inputs = make_synthetic_inputs(args, dtype=dtype, device=args.device, z_dim=z_dim)
    for k, v in inputs.items():
        dprint(f"input.{k}", value=v)

    # Stage 1: per-modality pre_dit (modality-specific stems)
    dsection("Stage 1: per-expert pre_dit (modality-specific stems)")
    video_pre = video_expert.pre_dit(
        x=inputs["video_latents"],
        timestep=inputs["timestep_video"],
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    track_pre = track_expert.pre_dit(
        x=inputs["track_latents"],
        timestep=inputs["timestep_track"],
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    action_pre = action_expert.pre_dit(
        action_tokens=inputs["action"],
        timestep=inputs["timestep_action"],
        context=inputs["context"],
        context_mask=inputs["context_mask"],
    )

    # Stage 2: MoT joint forward (SHARED mixed self-attention)
    dsection("Stage 2: MoT.forward (shared mixed self-attention across all experts)")
    attention_mask = build_attention_mask(
        video_seq_len=video_pre["tokens"].shape[1],
        track_seq_len=track_pre["tokens"].shape[1],
        action_seq_len=action_pre["tokens"].shape[1],
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        video_expert=video_expert,
        track_expert=track_expert,
        device=args.device,
    )
    dprint(
        "joint_mask",
        attention_mask=attention_mask,
        video_seq_len=video_pre["tokens"].shape[1],
        track_seq_len=track_pre["tokens"].shape[1],
        action_seq_len=action_pre["tokens"].shape[1],
    )
    tokens_out = mot(
        embeds_all={"video": video_pre["tokens"], "track": track_pre["tokens"], "action": action_pre["tokens"]},
        attention_mask=attention_mask,
        freqs_all={"video": video_pre["freqs"], "track": track_pre["freqs"], "action": action_pre["freqs"]},
        context_all={
            "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
            "track": {"context": track_pre["context"], "mask": track_pre["context_mask"]},
            "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
        },
        t_mod_all={
            "video": video_pre["t_mod"],
            "track": track_pre["t_mod"],
            "action": action_pre["t_mod"],
        },
    )

    # Stage 3: per-modality post_dit (modality-specific heads)
    dsection("Stage 3: per-expert post_dit (modality-specific heads)")
    pred_video = video_expert.post_dit(tokens_out["video"], video_pre)
    pred_track = track_expert.post_dit(tokens_out["track"], track_pre)
    pred_action = action_expert.post_dit(tokens_out["action"], action_pre)

    dsection("Final per-modality predictions")
    dprint(
        "predictions",
        pred_video=pred_video,
        pred_track=pred_track,
        pred_action=pred_action,
        match_video=tuple(pred_video.shape) == tuple(inputs["video_latents"].shape),
        match_track=tuple(pred_track.shape) == tuple(inputs["track_latents"].shape),
        match_action=tuple(pred_action.shape) == tuple(inputs["action"].shape),
    )
    print("\nDone. Predictions match input shapes? "
          f"video={tuple(pred_video.shape) == tuple(inputs['video_latents'].shape)}, "
          f"track={tuple(pred_track.shape) == tuple(inputs['track_latents'].shape)}, "
          f"action={tuple(pred_action.shape) == tuple(inputs['action'].shape)}")

    # ----- (optional) Symmetric VAE decode -----
    if vae is not None:
        dsection("Stage 4: VAE decode latent-space predictions back to pixels")
        # Random-init weights mean these "pixels" are garbage, but the *shape*
        # round-trip is the point: latent -> RGB tensor [B, 3, T_pix, H_pix, W_pix].
        decoded_video = vae.decode(pred_video, device=args.device)
        decoded_track = vae.decode(pred_track, device=args.device)
        dprint(
            "VAE.decoded",
            decoded_video=decoded_video,
            decoded_track=decoded_track,
        )


if __name__ == "__main__":
    main()
