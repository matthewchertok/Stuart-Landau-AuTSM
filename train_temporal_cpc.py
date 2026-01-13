#!/usr/bin/env python3
"""
Train a CPC-style temporal model on Stuart-Landau spatial latents.

This mirrors the double autoencoder pipeline but replaces the temporal stage
with a segment-level CPC objective inspired by
four_state_polymer_contrastive_AE/autoencoder.py: a temporal CNN encodes fixed
windows of spatial latents, a transformer context model predicts the correct
next window embedding among in-batch candidates, and the final context vector
optionally reconstructs the full latent sequence.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPC temporal model for spatial Stuart-Landau latents.")
    parser.add_argument(
        "--latent_csv",
        type=str,
        default="spatial_latent_vectors.csv",
        help="CSV produced by train_spatial.py containing per-frame latents.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where checkpoints, logs, and outputs are saved.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Trajectories per batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--max_epochs", type=int, default=400, help="Maximum training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--cpc_weight",
        type=float,
        default=1.0,
        help="Weight applied to the segment-level CPC loss.",
    )
    parser.add_argument(
        "--recon_weight",
        type=float,
        default=1.0,
        help="Weight applied to reconstructing the full latent trajectory.",
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=128,
        help="Base channel width for the temporal CNN.",
    )
    parser.add_argument(
        "--context_heads",
        type=int,
        default=4,
        help="Number of attention heads in the transformer context model.",
    )
    parser.add_argument(
        "--context_layers",
        type=int,
        default=2,
        help="Number of transformer encoder layers in the context model.",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=64,
        help="Window length (frames) for segment-level contrastive prediction.",
    )
    parser.add_argument(
        "--segment_stride",
        type=int,
        default=8,
        help="Stride (frames) between consecutive segments.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for InfoNCE logits in CPC.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for segment collation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string. Defaults to CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--include_reversed_trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Duplicate every trajectory with reversed time order before training.",
    )
    parser.add_argument(
        "--include_scrambled_trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Duplicate each original trajectory with a randomly shuffled time ordering.",
    )
    parser.add_argument(
        "--scrambled_seed",
        type=int,
        default=None,
        help="Seed used when shuffling scrambled duplicates (defaults to --seed).",
    )
    parser.add_argument(
        "--scramble_order",
        action="store_true",
        help="Run an additional experiment where every trajectory's time indices are randomized.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class LatentTrajectory:
    ensemble_number: int
    k_value: float
    stack: np.ndarray  # (seq_len, latent_dim)
    order_label: str
    traj_id: str


def load_latent_stacks(csv_path: str) -> List[LatentTrajectory]:
    rows: Dict[Tuple[int, float], List[Tuple[float, List[float]]]] = {}
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        latent_columns = [col for col in (reader.fieldnames or []) if col.startswith("latent_")]
        if not latent_columns:
            raise RuntimeError("No latent_* columns found in the provided CSV.")
        for row in reader:
            ensemble = int(row["ensemble_number"])
            k_value = float(row["K_value"])
            time_value = float(row["t"])
            latent_values = [float(row[col]) for col in latent_columns]
            key = (ensemble, k_value)
            rows.setdefault(key, []).append((time_value, latent_values))

    trajectories: List[LatentTrajectory] = []
    for idx, ((ensemble, k_value), entries) in enumerate(sorted(rows.items())):
        entries.sort(key=lambda item: item[0])
        stack = np.asarray([latent for _, latent in entries], dtype=np.float32)
        traj_id = f"ensemble{ensemble}_K{k_value:.6g}_idx{idx}"
        trajectories.append(
            LatentTrajectory(
                ensemble_number=ensemble,
                k_value=k_value,
                stack=stack,
                order_label="forward",
                traj_id=traj_id,
            )
        )
    return trajectories


def scramble_trajectories(
    trajectories: Sequence[LatentTrajectory],
    seed: int,
    order_label_override: str | None = None,
    suffix: str = "scrambled",
) -> List[LatentTrajectory]:
    rng = np.random.default_rng(seed)
    scrambled: List[LatentTrajectory] = []
    for traj in trajectories:
        seq_len = traj.stack.shape[0]
        perm = rng.permutation(seq_len)
        scrambled_stack = traj.stack[perm].copy()
        scrambled.append(
            LatentTrajectory(
                ensemble_number=traj.ensemble_number,
                k_value=traj.k_value,
                stack=scrambled_stack,
                order_label=order_label_override or traj.order_label,
                traj_id=f"{traj.traj_id}_{suffix}",
            )
        )
    return scrambled


def reverse_trajectories(trajectories: Sequence[LatentTrajectory]) -> List[LatentTrajectory]:
    reversed_traj: List[LatentTrajectory] = []
    for traj in trajectories:
        reversed_traj.append(
            LatentTrajectory(
                ensemble_number=traj.ensemble_number,
                k_value=traj.k_value,
                stack=traj.stack[::-1].copy(),
                order_label="reversed",
                traj_id=f"{traj.traj_id}_rev",
            )
        )
    return reversed_traj


class SegmentSequenceDataset(Dataset):
    """Emit sequences of fixed-length latent segments for CPC training."""

    def __init__(
        self,
        trajectories: Sequence[LatentTrajectory],
        segment_length: int,
        segment_stride: int,
    ) -> None:
        self.segment_length = int(segment_length)
        self.segment_stride = int(segment_stride)
        self.items: List[Tuple[List[np.ndarray], LatentTrajectory]] = []
        for traj in trajectories:
            seq = traj.stack
            length = seq.shape[0]
            start = 0
            segments: List[np.ndarray] = []
            while start + self.segment_length <= length:
                segments.append(seq[start : start + self.segment_length].copy())
                start += self.segment_stride
            if segments:
                self.items.append((segments, traj))
        if not self.items:
            raise RuntimeError(
                "No segment sequences could be formed; adjust segment_length or segment_stride."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        segments, traj = self.items[idx]
        torch_segments = [torch.from_numpy(segment).float().transpose(0, 1) for segment in segments]
        return torch_segments, traj.traj_id


def collate_segment_sequences(batch, sequence_by_id: Dict[str, np.ndarray]):
    segment_lists, traj_ids = zip(*batch)
    lengths = torch.tensor([len(segments) for segments in segment_lists], dtype=torch.long)
    max_segments = int(lengths.max().item())
    if max_segments == 0:
        raise ValueError("No segments in batch.")
    segment_length = segment_lists[0][0].shape[-1]
    latent_dim = segment_lists[0][0].shape[0]
    padded = torch.zeros(len(segment_lists), max_segments, latent_dim, segment_length, dtype=torch.float32)
    segment_mask = torch.zeros(len(segment_lists), max_segments, dtype=torch.bool)
    for i, segments in enumerate(segment_lists):
        count = len(segments)
        if count == 0:
            continue
        padded[i, :count] = torch.stack(segments, dim=0)
        segment_mask[i, :count] = True

    full_seqs = [torch.from_numpy(sequence_by_id[tid]).float() for tid in traj_ids]
    frame_lengths = torch.tensor([seq.shape[0] for seq in full_seqs], dtype=torch.long)
    max_frames = int(frame_lengths.max().item())
    full_padded = torch.zeros(len(full_seqs), max_frames, latent_dim, dtype=torch.float32)
    for i, seq in enumerate(full_seqs):
        full_padded[i, : seq.shape[0]] = seq
    return padded, segment_mask, lengths, list(traj_ids), full_padded, frame_lengths


def conv_output_length(length: torch.Tensor, kernel: int, stride: int, padding: int) -> torch.Tensor:
    """Match PyTorch Conv1d output length calculation."""
    return torch.div(length + 2 * padding - (kernel - 1) - 1, stride, rounding_mode="floor") + 1


class TemporalConvAutoencoder(nn.Module):
    """Strided temporal CNN encoder with an upsampling decoder."""

    def __init__(self, latent_dim: int = 16, hidden_channels: int = 128, input_channels: int = 16):
        super().__init__()
        hidden_channels = max(hidden_channels, 32)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels * 2, latent_dim, kernel_size=1, stride=1, padding=0),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, input_channels, kernel_size=1),
        )
        self.latent_dim = latent_dim
        self.input_channels = input_channels

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor, target_len: int) -> torch.Tensor:
        recon = self.decoder(latents)
        if recon.size(-1) > target_len:
            recon = recon[..., :target_len]
        elif recon.size(-1) < target_len:
            pad_len = target_len - recon.size(-1)
            recon = F.pad(recon, (0, pad_len))
        return recon

    def forward(self, inputs: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(inputs)
        recon = self.decode(latents, target_len)
        return recon, latents

    def downsample_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = conv_output_length(lengths, kernel=5, stride=2, padding=2)
        out = conv_output_length(out, kernel=5, stride=2, padding=2)
        return torch.clamp(out, min=1)


class ContextTrajectoryDecoder(nn.Module):
    """Decode the final context embedding into a full latent trajectory."""

    def __init__(self, context_dim: int, latent_dim: int, hidden_size: int = 256, max_len: int = 10_000) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, context_dim)
        self.init_proj = nn.Linear(context_dim, hidden_size)
        self.gru = nn.GRU(input_size=context_dim, hidden_size=hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, context: torch.Tensor, target_len: int) -> torch.Tensor:
        bsz, _ = context.shape
        positions = torch.arange(target_len, device=context.device)
        pos_tokens = self.pos_embedding(positions).unsqueeze(0).expand(bsz, -1, -1)
        h0 = self.init_proj(context).unsqueeze(0)  # (1, B, hidden_size)
        outputs, _ = self.gru(pos_tokens, h0)
        return self.output_proj(outputs)  # (B, target_len, latent_dim)


class SegmentContextModel(nn.Module):
    """Transformer encoder that builds autoregressive segment context."""

    def __init__(
        self,
        embed_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 10_000, embed_dim))
        self.embed_dim = embed_dim

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = embeddings.shape
        if self.pos_embedding.size(1) < seq_len:
            raise ValueError(f"pos_embedding too short for seq_len={seq_len}")
        pos = self.pos_embedding[:, :seq_len, :]
        x = embeddings + pos
        causal_mask = torch.ones(seq_len, seq_len, device=embeddings.device, dtype=torch.bool).triu(1)
        key_padding_mask = ~mask
        context = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return context


def train_one_epoch(
    model: TemporalConvAutoencoder,
    context_model: SegmentContextModel,
    trajectory_decoder: ContextTrajectoryDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cpc_weight: float,
    recon_weight: float,
    temperature: float,
) -> Dict[str, float]:
    model.train()
    context_model.train()
    totals = {"total": 0.0, "cpc": 0.0, "recon": 0.0}
    batches = 0
    for batch in loader:
        segments, segment_mask, lengths, traj_ids, full_trajs, frame_lengths = batch
        segments = segments.to(device)
        segment_mask = segment_mask.to(device)
        frame_lengths = frame_lengths.to(device)
        full_trajs = full_trajs.to(device)
        bsz, max_seg, latent_dim, seg_len = segments.shape
        optimizer.zero_grad(set_to_none=True)
        flat_segments = segments.view(bsz * max_seg, latent_dim, seg_len)
        segment_latents = model.encode(flat_segments)
        latent_seq_len = segment_latents.shape[-1]
        segment_latents = segment_latents.view(bsz, max_seg, model.latent_dim, latent_seq_len)
        segment_embeddings = segment_latents.mean(dim=-1)
        context = context_model(segment_embeddings, segment_mask)
        indices = segment_mask.sum(dim=1) - 1
        indices = torch.clamp(indices, min=0)
        final_ctx = context[torch.arange(bsz, device=device), indices]
        if max_seg > 1:
            ctx = context[:, :-1, :]
            tgt = segment_embeddings[:, 1:, :]
            valid = segment_mask[:, :-1] & segment_mask[:, 1:]
            if valid.any():
                ctx_flat = ctx[valid]
                tgt_flat = tgt[valid]
                logits = (ctx_flat @ tgt_flat.T) / max(temperature, 1e-4)
                targets = torch.arange(logits.size(0), device=device)
                cpc_loss = F.cross_entropy(logits, targets)
            else:
                cpc_loss = segment_latents.new_tensor(0.0)
        else:
            cpc_loss = segment_latents.new_tensor(0.0)

        if recon_weight > 0.0:
            max_frames = int(frame_lengths.max().item())
            recon = trajectory_decoder(final_ctx, target_len=max_frames)  # (B, T_max, latent_dim)
            target = full_trajs  # (B, T_max, latent_dim)
            time_mask = torch.arange(max_frames, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
            time_mask = time_mask.unsqueeze(-1)  # (B, T_max, 1)
            mse = F.mse_loss(recon, target, reduction="none")
            mse = (mse * time_mask).sum() / time_mask.sum().clamp_min(1.0)
            recon_loss = mse
        else:
            recon_loss = segment_latents.new_tensor(0.0)

        loss = cpc_weight * cpc_loss + recon_weight * recon_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(context_model.parameters()), 1.0)
        optimizer.step()
        totals["total"] += loss.item()
        totals["cpc"] += cpc_loss.item()
        totals["recon"] += recon_loss.item()
        batches += 1

    denom = max(batches, 1)
    for key in totals:
        totals[key] /= denom
    return totals


def encode_context_latents(
    model: TemporalConvAutoencoder,
    context_model: SegmentContextModel,
    trajectories: Sequence[LatentTrajectory],
    segment_length: int,
    segment_stride: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    context_model.eval()
    latents: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(trajectories), batch_size):
            batch_traj = trajectories[start : start + batch_size]
            segment_lists: List[torch.Tensor] = []
            seg_counts: List[int] = []
            for traj in batch_traj:
                segments: List[torch.Tensor] = []
                idx = 0
                while idx + segment_length <= traj.stack.shape[0]:
                    seg_np = traj.stack[idx : idx + segment_length]
                    segments.append(torch.from_numpy(seg_np).float().transpose(0, 1))
                    idx += segment_stride
                if not segments:
                    continue
                segment_lists.append(torch.stack(segments, dim=0))
                seg_counts.append(len(segments))
            if not segment_lists:
                continue
            max_seg = max(seg_counts)
            seg_len = segment_lists[0].shape[-1]
            latent_dim = segment_lists[0].shape[1]
            batch_size_eff = len(segment_lists)
            padded = torch.zeros(batch_size_eff, max_seg, latent_dim, seg_len, device=device)
            mask = torch.zeros(batch_size_eff, max_seg, dtype=torch.bool, device=device)
            for i, segs in enumerate(segment_lists):
                count = segs.shape[0]
                padded[i, :count] = segs.to(device)
                mask[i, :count] = True
            flat = padded.view(batch_size_eff * max_seg, latent_dim, seg_len)
            segment_latents = model.encode(flat)
            latent_seq_len = segment_latents.shape[-1]
            segment_latents = segment_latents.view(batch_size_eff, max_seg, model.latent_dim, latent_seq_len)
            segment_embeddings = segment_latents.mean(dim=-1)
            context = context_model(segment_embeddings, mask)
            indices = mask.sum(dim=1) - 1
            indices = torch.clamp(indices, min=0)
            final_ctx = context[torch.arange(batch_size_eff, device=device), indices]
            latents.append(final_ctx.cpu().numpy())
    return np.concatenate(latents, axis=0) if latents else np.zeros((0, model.latent_dim), dtype=np.float32)


def save_training_log(path: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "cpc", "weighted_cpc", "recon", "weighted_recon", "total"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_spatiotemporal_latents(path: Path, trajectories: Sequence[LatentTrajectory], latents: np.ndarray) -> None:
    latent_dim = latents.shape[-1]
    fieldnames = ["ensemble_number", "K_value", "order_label"] + [
        f"spatiotemporal_latent_{i}" for i in range(latent_dim)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for traj, latent_vec in zip(trajectories, latents):
            row = {
                "ensemble_number": traj.ensemble_number,
                "K_value": traj.k_value,
                "order_label": traj.order_label,
            }
            for i, value in enumerate(latent_vec):
                row[f"spatiotemporal_latent_{i}"] = float(value)
            writer.writerow(row)


def run_training_pipeline(
    args: argparse.Namespace,
    trajectories: Sequence[LatentTrajectory],
    output_dir: Path,
    device: torch.device,
    run_suffix: str,
) -> None:
    prefix = f"[{run_suffix}] "
    segment_dataset = SegmentSequenceDataset(
        trajectories=trajectories,
        segment_length=args.segment_length,
        segment_stride=args.segment_stride,
    )
    sequence_by_id: Dict[str, np.ndarray] = {traj.traj_id: traj.stack for traj in trajectories}

    def _collate(batch):
        return collate_segment_sequences(batch, sequence_by_id=sequence_by_id)

    loader = DataLoader(
        segment_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
        persistent_workers=args.num_workers > 0,
    )

    latent_dim = trajectories[0].stack.shape[1]
    model = TemporalConvAutoencoder(
        latent_dim=latent_dim,
        hidden_channels=args.hidden_channels,
        input_channels=latent_dim,
    ).to(device)
    context_model = SegmentContextModel(
        embed_dim=model.latent_dim,
        nhead=args.context_heads,
        num_layers=args.context_layers,
    ).to(device)
    trajectory_decoder = ContextTrajectoryDecoder(
        context_dim=context_model.embed_dim, latent_dim=latent_dim, hidden_size=args.hidden_channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(context_model.parameters()) + list(trajectory_decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_total = float("inf")
    stagnant = 0

    for epoch in range(1, args.max_epochs + 1):
        metrics = train_one_epoch(
            model=model,
            context_model=context_model,
            trajectory_decoder=trajectory_decoder,
            loader=loader,
            optimizer=optimizer,
            device=device,
            cpc_weight=args.cpc_weight,
            recon_weight=args.recon_weight,
            temperature=args.temperature,
        )
        weighted_cpc = metrics["cpc"] * args.cpc_weight
        weighted_recon = metrics["recon"] * args.recon_weight
        total = weighted_cpc + weighted_recon
        history.append(
            {
                "epoch": epoch,
                "cpc": metrics["cpc"],
                "weighted_cpc": weighted_cpc,
                "recon": metrics["recon"],
                "weighted_recon": weighted_recon,
                "total": total,
            }
        )
        improved = total + 1e-6 < best_total
        if improved:
            best_total = total
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "context": copy.deepcopy(context_model.state_dict()),
                "decoder": copy.deepcopy(trajectory_decoder.state_dict()),
            }
            stagnant = 0
        else:
            stagnant += 1
        print(
            prefix
            + f"Epoch {epoch:04d} total={total:.4f} recon(w={args.recon_weight:.2f})={weighted_recon:.4f} "
            + f"cpc(w={args.cpc_weight:.2f})={weighted_cpc:.4f} "
            + ("*" if improved else ""),
            flush=True,
        )
        if stagnant >= args.patience:
            print(prefix + f"No improvement for {args.patience} epochs; stopping early at epoch {epoch}.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        context_model.load_state_dict(best_state["context"])
        trajectory_decoder.load_state_dict(best_state["decoder"])

    checkpoint_path = output_dir / f"temporal_cpc_{run_suffix}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": {"latent_dim": model.latent_dim, "hidden_channels": args.hidden_channels, "input_channels": latent_dim},
            "context_state_dict": context_model.state_dict(),
            "context_kwargs": {
                "embed_dim": model.latent_dim,
                "nhead": args.context_heads,
                "num_layers": args.context_layers,
                "dim_feedforward": 256,
                "dropout": 0.1,
            },
            "decoder_state_dict": trajectory_decoder.state_dict(),
            "decoder_kwargs": {
                "context_dim": model.latent_dim,
                "latent_dim": latent_dim,
                "hidden_size": args.hidden_channels,
                "max_len": 10_000,
            },
            "segment_length": args.segment_length,
            "segment_stride": args.segment_stride,
        },
        checkpoint_path,
    )
    print(prefix + f"Saved best checkpoint to {checkpoint_path}", flush=True)

    history_path = output_dir / f"temporal_cpc_training_log_{run_suffix}.csv"
    save_training_log(history_path, history)
    print(prefix + f"Saved training log to {history_path}", flush=True)

    latents = encode_context_latents(
        model=model,
        context_model=context_model,
        trajectories=trajectories,
        segment_length=args.segment_length,
        segment_stride=args.segment_stride,
        batch_size=args.batch_size,
        device=device,
    )
    latent_csv = output_dir / f"spatiotemporal_latent_vectors_{run_suffix}.csv"
    save_spatiotemporal_latents(latent_csv, trajectories, latents)
    print(prefix + f"Saved spatiotemporal latents to {latent_csv}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_latent_stacks(args.latent_csv)
    if not trajectories:
        raise RuntimeError(f"No latent stacks found in {args.latent_csv}")
    base_traj = list(trajectories)
    if args.include_reversed_trajectories:
        reversed_traj = reverse_trajectories(base_traj)
        base_traj.extend(reversed_traj)
        print(f"Added {len(reversed_traj)} reversed trajectories (total {len(base_traj)})", flush=True)
    if args.include_scrambled_trajectories:
        scramble_seed = args.scrambled_seed if args.scrambled_seed is not None else args.seed
        scrambled = scramble_trajectories(base_traj, scramble_seed, order_label_override="scrambled")
        base_traj.extend(scrambled)
        print(f"Added {len(scrambled)} scrambled trajectories with seed={scramble_seed} (total {len(base_traj)})", flush=True)

    experiments: List[Tuple[str, Sequence[LatentTrajectory]]] = [("correct_order", base_traj)]
    if args.scramble_order:
        scrambled_full = scramble_trajectories(base_traj, args.seed, suffix="order_shuffle", order_label_override="scrambled")
        experiments.append(("scrambled", scrambled_full))

    for run_suffix, trajs in experiments:
        print(f"\n=== Training run: {run_suffix} ===", flush=True)
        run_training_pipeline(args, trajs, output_dir, device, run_suffix)


if __name__ == "__main__":
    main()
