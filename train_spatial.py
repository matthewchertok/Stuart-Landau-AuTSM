#!/usr/bin/env python3
"""
Train a transformer-based spatial autoencoder that treats oscillators as tokens
and summarizes every frame with a CLS latent. Each frame is reconstructed solely
from the CLS latent and input-independent oscillator queries, enforcing a strict
bottleneck before producing per-frame 16D latents for downstream temporal models.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


DEFAULT_SPATIAL_DATA_CSV = "stuart_landau_trajectories_with_replicates.csv"
REPLICATE_DATA_CSV = "stuart_landau_trajectories_with_replicates.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CLS-pooled transformer autoencoder over oscillators."
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=DEFAULT_SPATIAL_DATA_CSV,
        help="Fallback CSV file that contains Stuart-Landau trajectories.",
    )
    parser.add_argument(
        "--use_replicate_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the consolidated replicate dataset instead of --data_csv.",
    )
    parser.add_argument(
        "--replicate_data_csv",
        type=str,
        default=REPLICATE_DATA_CSV,
        help="Path to the replicate CSV that replaces --data_csv when --use_replicate_dataset is enabled.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where checkpoints, logs, and latents are saved.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Number of trajectories per batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--max_epochs", type=int, default=250, help="Maximum training epochs.")
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.005,
        help="Relative validation loss improvement required to reset patience.",
    )
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--model_dim", type=int, default=128, help="Transformer hidden size.")
    parser.add_argument("--nhead", type=int, default=8, help="Multi-head attention heads.")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Encoder depth.")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Decoder depth.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout inside transformer layers.")
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Dimensionality of the per-frame spatial latent vector.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier (defaults to CUDA if available).",
    )
    parser.add_argument(
        "--test_recon_quality",
        action="store_true",
        help="Train for a single epoch and export original/reconstruction videos.",
    )
    parser.add_argument(
        "--test_recon_cases",
        type=int,
        default=3,
        help="Number of trajectories to visualize when --test_recon_quality is enabled.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrajectoryRecord:
    ensemble_number: int
    k_value: float
    times: List[float]
    frames: np.ndarray  # (seq_len, num_osc, 2)


def load_trajectories(csv_path: str) -> List[TrajectoryRecord]:
    """
    Load trajectories while preserving oscillator tokens per frame.
    Returns frames with shape (seq_len, num_osc, 2).
    """
    frame_store: Dict[Tuple[int, float], Dict[float, Dict[int, Tuple[float, float]]]] = {}
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ensemble = int(row["ensemble_number"])
            k_value = float(row["K_value"])
            osc = int(row["oscillator_number"])
            real = float(row["real"])
            imag = float(row["imaginary"])
            time_value = float(row["t"])
            key = (ensemble, k_value)
            if key not in frame_store:
                frame_store[key] = {}
            if time_value not in frame_store[key]:
                frame_store[key][time_value] = {}
            frame_store[key][time_value][osc] = (real, imag)

    records: List[TrajectoryRecord] = []
    for (ensemble, k_value), time_dict in sorted(frame_store.items()):
        times_sorted = sorted(time_dict.keys())
        frame_list: List[List[List[float]]] = []
        osc_indices = sorted(next(iter(time_dict.values())).keys())
        for t in times_sorted:
            osc_dict = time_dict[t]
            frame_tokens: List[List[float]] = []
            for osc in osc_indices:
                real, imag = osc_dict[osc]
                frame_tokens.append([real, imag])
            frame_list.append(frame_tokens)
        frames = np.asarray(frame_list, dtype=np.float32)  # (seq_len, num_osc, 2)
        records.append(
            TrajectoryRecord(
                ensemble_number=ensemble,
                k_value=k_value,
                times=times_sorted,
                frames=frames,
            )
        )
    return records


class TrajectoryDataset(Dataset):
    def __init__(self, records: Sequence[TrajectoryRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.records[idx].frames)  # (seq, num_osc, 2)


class FrameCLSAutoencoder(nn.Module):
    def __init__(
        self,
        num_oscillators: int,
        features_per_oscillator: int,
        latent_dim: int,
        model_dim: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_osc = num_oscillators
        self.features_per_token = features_per_oscillator
        self.latent_dim = latent_dim

        self.token_proj = nn.Linear(features_per_oscillator, model_dim)
        self.osc_pos = nn.Parameter(torch.randn(num_oscillators, model_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.to_latent = nn.Linear(model_dim, latent_dim)

        self.latent_to_memory = nn.Linear(latent_dim, model_dim)
        self.decoder_queries = nn.Parameter(torch.randn(num_oscillators, model_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(model_dim, features_per_oscillator)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (batch, seq_len, num_osc, features)
        batch, seq_len, num_osc, feat = frames.shape
        assert num_osc == self.num_osc
        tokens = frames.view(batch * seq_len, num_osc, feat)
        tokens = self.token_proj(tokens) + self.osc_pos.unsqueeze(0)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        encoder_input = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(encoder_input)
        cls_out = encoded[:, 0, :]
        latents = self.to_latent(cls_out)
        return latents.view(batch, seq_len, self.latent_dim)

    def decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: (batch, seq_len, latent_dim)
        batch, seq_len, latent_dim = latents.shape
        assert latent_dim == self.latent_dim
        flattened = latents.view(batch * seq_len, latent_dim)
        memory = self.latent_to_memory(flattened).unsqueeze(1)
        queries = self.decoder_queries.unsqueeze(0).expand(flattened.size(0), -1, -1)
        decoded = self.decoder(queries, memory)
        outputs = self.output_proj(decoded)
        outputs = outputs.view(batch, seq_len, self.num_osc, self.features_per_token)
        return outputs

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(frames)

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode_frames(frames)
        recon = self.decode_frames(latents)
        return recon, latents


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_one_epoch(
    model: FrameCLSAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: FrameCLSAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for batch in loader:
        batch = batch.to(device)
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


def save_history(history: List[Dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_latent_csv(
    path: Path,
    records: Sequence[TrajectoryRecord],
    latents: Sequence[np.ndarray],
) -> None:
    latent_dim = latents[0].shape[-1]
    fieldnames = ["ensemble_number", "K_value", "t"] + [f"latent_{i}" for i in range(latent_dim)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, latent_stack in zip(records, latents):
            for t_value, latent_vec in zip(record.times, latent_stack):
                row = {
                    "ensemble_number": record.ensemble_number,
                    "K_value": record.k_value,
                    "t": t_value,
                }
                for i, value in enumerate(latent_vec):
                    row[f"latent_{i}"] = float(value)
                writer.writerow(row)


def apply_inverse_transformations(frames: np.ndarray) -> np.ndarray:
    """
    Revert any preprocessing applied before training.

    Currently, frames are fed to the model in their original real/imag form, so
    the inverse transform is simply an identity copy. A dedicated helper keeps
    the logic in one place in case preprocessing is introduced later.
    """
    return np.asarray(frames, dtype=np.float32)


def select_trajectories_for_testing(
    records: Sequence[TrajectoryRecord], limit: int
) -> List[TrajectoryRecord]:
    if limit <= 0:
        return []
    if limit >= len(records):
        return list(records)
    indices = sorted(set(np.linspace(0, len(records) - 1, num=limit, dtype=int).tolist()))
    return [records[idx] for idx in indices]


@torch.no_grad()
def reconstruct_frames(
    model: FrameCLSAutoencoder, frames: np.ndarray, device: torch.device
) -> np.ndarray:
    tensor = torch.from_numpy(frames).unsqueeze(0).to(device)
    latents = model.encode(tensor)
    recon = model.decode_frames(latents).squeeze(0).cpu().numpy()
    return recon.astype(np.float32)


def expand_range(min_val: float, max_val: float, min_width: float = 1e-3) -> Tuple[float, float]:
    span = max_val - min_val
    if span < min_width or math.isclose(min_val, max_val):
        pad = max(min_width, abs(min_val) * 0.05 + min_width)
        return min_val - pad, max_val + pad
    pad = span * 0.05
    return min_val - pad, max_val + pad


def compute_axis_limits(
    stacks: Sequence[np.ndarray], times: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    real_vals = [stack[..., 0] for stack in stacks]
    imag_vals = [stack[..., 1] for stack in stacks]
    min_real = float(min(np.min(vals) for vals in real_vals))
    max_real = float(max(np.max(vals) for vals in real_vals))
    min_imag = float(min(np.min(vals) for vals in imag_vals))
    max_imag = float(max(np.max(vals) for vals in imag_vals))
    if times.size == 0:
        raise ValueError("Cannot build videos without timestamps.")
    min_t = float(times.min())
    max_t = float(times.max())
    real_min, real_max = expand_range(min_real, max_real)
    imag_min, imag_max = expand_range(min_imag, max_imag)
    time_min, time_max = expand_range(min_t, max_t)
    return real_min, real_max, imag_min, imag_max, time_min, time_max


def render_trajectory_video(
    frames: np.ndarray,
    times: np.ndarray,
    output_path: Path,
    title: str,
    axis_limits: Tuple[float, float, float, float, float, float],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ensure_dir(output_path.parent)
    seq_len, num_osc, _ = frames.shape
    if seq_len == 0 or num_osc == 0:
        raise ValueError("Cannot render an empty trajectory.")
    real = frames[..., 0]
    imag = frames[..., 1]
    min_real, max_real, min_imag, max_imag, min_t, max_t = axis_limits
    times = np.asarray(times, dtype=np.float32)
    if times.shape[0] != seq_len:
        raise ValueError("Number of timestamps and frames must match for video export.")

    start_color = np.array([0x12, 0x12, 0x12], dtype=np.float32) / 255.0
    end_color = np.array([0xE7, 0x75, 0x00], dtype=np.float32) / 255.0
    if num_osc > 1:
        color_ratios = np.linspace(0.0, 1.0, num=num_osc)
    else:
        color_ratios = np.array([0.0], dtype=np.float32)
    colors = [
        tuple(start_color * (1.0 - ratio) + end_color * ratio) for ratio in color_ratios
    ]

    fig = plt.figure(figsize=(2.5, 2.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25.0, azim=-60.0)

    def update(frame_idx: int) -> List:
        ax.clear()
        ax.set_xlim(min_real, max_real)
        ax.set_ylim(min_imag, max_imag)
        ax.set_zlim(min_t, max_t)
        ax.set_xlabel("Re", labelpad=-10)
        ax.set_ylabel("Im", labelpad=-10)
        ax.set_zlabel("t", labelpad=-16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        upto = frame_idx + 1
        current_time = times[min(frame_idx, seq_len - 1)]
        for osc_idx in range(num_osc):
            color = colors[osc_idx]
            ax.plot(
                real[:upto, osc_idx],
                imag[:upto, osc_idx],
                times[:upto],
                lw=0.5,
                color=color,
                antialiased=False,
            )
        ax.set_title(f"{title} | t={current_time:.2f}", fontsize=8)
        return []

    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=50, blit=False)
    writer = animation.FFMpegWriter(fps=24, bitrate=2000)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)


def export_reconstruction_videos(
    model: FrameCLSAutoencoder,
    records: Sequence[TrajectoryRecord],
    device: torch.device,
    output_dir: Path,
    max_cases: int,
) -> None:
    selected = select_trajectories_for_testing(records, max_cases)
    if not selected:
        print("No trajectories selected for reconstruction videos.", flush=True)
        return

    ensure_dir(output_dir)
    model.eval()
    for record in selected:
        recon = reconstruct_frames(model, record.frames, device)
        original = apply_inverse_transformations(record.frames)
        reconstructed = apply_inverse_transformations(recon)
        times = np.asarray(record.times, dtype=np.float32)
        axis_limits = compute_axis_limits([original, reconstructed], times)

        subdir = output_dir / f"K_{record.k_value:.3f}_ensemble_{record.ensemble_number:03d}"
        ensure_dir(subdir)
        print(
            f"Saving reconstruction videos for ensemble {record.ensemble_number} "
            f"(K={record.k_value:.3f}) to {subdir}",
            flush=True,
        )
        render_trajectory_video(
            original,
            times,
            subdir / "original.mp4",
            f"Original K={record.k_value:.3f}",
            axis_limits,
        )
        render_trajectory_video(
            reconstructed,
            times,
            subdir / "reconstruction.mp4",
            f"Reconstruction K={record.k_value:.3f}",
            axis_limits,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.test_recon_quality and args.test_recon_cases <= 0:
        raise ValueError("--test_recon_cases must be positive when --test_recon_quality is set.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}", flush=True)

    data_csv_path = args.replicate_data_csv if args.use_replicate_dataset else args.data_csv
    print(f"Loading trajectories from {data_csv_path}", flush=True)
    records = load_trajectories(data_csv_path)
    if not records:
        raise RuntimeError(f"No trajectories found in {data_csv_path}")

    dataset = TrajectoryDataset(records)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    seq_len, num_osc, feat_dim = records[0].frames.shape
    model = FrameCLSAutoencoder(
        num_oscillators=num_osc,
        features_per_oscillator=feat_dim,
        latent_dim=args.latent_dim,
        model_dim=args.model_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}", flush=True)

        if not np.isfinite(best_val_loss):
            improved = True
        else:
            relative_improvement = (best_val_loss - val_loss) / max(best_val_loss, 1e-12)
            improved = relative_improvement > args.min_improvement

        if improved:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"  New best model with val_loss={best_val_loss:.6f}", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"  No significant improvement ({epochs_without_improvement}/{args.patience})", flush=True)

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.", flush=True)
            break

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "spatial_transformer.pt"
    torch.save({"model_state_dict": best_state, "config": vars(args)}, checkpoint_path)
    print(f"Saved best model to {checkpoint_path}", flush=True)

    history_path = output_dir / "spatial_training_log.csv"
    save_history(history, history_path)
    print(f"Saved training log to {history_path}", flush=True)

    # Export per-frame latents for every trajectory
    model.eval()
    latent_stacks: List[np.ndarray] = []
    with torch.no_grad():
        for record in records:
            frames = torch.from_numpy(record.frames).unsqueeze(0).to(device)
            latents = model.encode(frames).squeeze(0).cpu().numpy()
            latent_stacks.append(latents.astype(np.float32))

    latent_csv_path = output_dir / "spatial_latent_vectors.csv"
    save_latent_csv(latent_csv_path, records, latent_stacks)
    print(f"Saved latent vectors to {latent_csv_path}", flush=True)

    if args.test_recon_quality:
        video_dir = output_dir / "recon_quality_videos"
        export_reconstruction_videos(
            model=model,
            records=records,
            device=device,
            output_dir=video_dir,
            max_cases=args.test_recon_cases,
        )


if __name__ == "__main__":
    main()
