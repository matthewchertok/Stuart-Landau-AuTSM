#!/usr/bin/env python3
"""Project spatiotemporal latents with UMAP and color by order and K."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project spatiotemporal latents and create consistent UMAPs for order and K."
    )
    parser.add_argument(
        "--latent_csv",
        type=Path,
        default=Path("spatiotemporal_latent_vectors_correct_order.csv"),
        help="CSV produced by train_temporal.py that contains spatiotemporal latents with order labels.",
    )
    parser.add_argument(
        "--scrambled_latent_csv",
        type=Path,
        default=Path("spatiotemporal_latent_vectors_scrambled.csv"),
        help="CSV containing scrambled trajectories' spatiotemporal latents.",
    )
    parser.add_argument(
        "--include_correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to plot the standard (correct order) dataset.",
    )
    parser.add_argument(
        "--include_scrambled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to plot the fully scrambled dataset.",
    )
    parser.add_argument(
        "--order_output",
        type=Path,
        default=Path("outputs") / "spatiotemporal_latents_coloredby_order.png",
        help="Path for the Forward/Reversed/Scrambled UMAP figure.",
    )
    parser.add_argument(
        "--k_output",
        type=Path,
        default=Path("outputs") / "spatiotemporal_latents_coloredby_k.png",
        help="Path for the K-colored UMAP figure.",
    )
    parser.add_argument(
        "--scrambled_order_output",
        type=Path,
        default=Path("outputs") / "spatiotemporal_latents_scrambled_coloredby_order.png",
        help="Path for the fully scrambled dataset's order-colored UMAP.",
    )
    parser.add_argument(
        "--scrambled_k_output",
        type=Path,
        default=Path("outputs") / "spatiotemporal_latents_scrambled_coloredby_k.png",
        help="Path for the fully scrambled dataset's K-colored UMAP.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=None,
        help="Number of neighbors for the UMAP graph (defaults to len(latents) - 1).",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=1.0,
        help="Minimum distance between embedded points in UMAP.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP (use -1 for stochastic runs).",
    )
    parser.add_argument("--dpi", type=int, default=1000, help="Figure resolution.")
    parser.add_argument(
        "--title",
        type=str,
        default="Spatiotemporal Latents",
        help="Base title for the saved UMAP figures.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_order_label(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "order_label" in df.columns:
        return df
    total = len(df)
    if total == 0:
        raise RuntimeError(f"{path} is empty.")
    if total % 2 != 0:
        raise RuntimeError(
            f"{path} contains {total} rows; without 'order_label' we require an even number "
            "to infer forward versus reversed halves."
        )
    half = total // 2
    print(
        f"Adding inferred order labels to {path} (first {half} forward, last {total - half} reversed)",
        flush=True,
    )
    df = df.copy()
    df["order_label"] = ["forward"] * half + ["reversed"] * (total - half)
    return df


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return _ensure_order_label(df, path)


def _load_dataset(name: str, path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skipping {name} dataset because {path} is missing.", flush=True)
        return None
    df = _load_dataframe(path)
    print(f"Loaded {len(df)} rows from {path}", flush=True)
    return df


def _latent_columns(df: pd.DataFrame) -> Sequence[str]:
    columns = sorted(
        (col for col in df.columns if col.startswith("spatiotemporal_latent_")),
        key=lambda value: int(value.split("_")[-1]),
    )
    if not columns:
        raise RuntimeError(
            "Latent column names prefixed with 'spatiotemporal_latent_' were not found."
        )
    return columns


def _compute_embedding(
    latents: np.ndarray,
    n_neighbors: int | None,
    min_dist: float,
    seed: int | None,
) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("UMAP is required to create the projection.") from exc

    max_neighbors = max(2, len(latents) - 1)
    neighbors = n_neighbors if n_neighbors is not None else max_neighbors
    neighbors = min(neighbors, max_neighbors)
    random_state = None if seed is None or seed < 0 else seed
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        n_jobs=1,
    )
    return reducer.fit_transform(latents)


def _plot_order(
    embedding: np.ndarray,
    order_labels: Sequence[str],
    path: Path,
    dpi: int,
) -> None:
    color_map = {"forward": "#e77500", "reversed": "#121212", "scrambled": "#0021a5"}
    fig, ax = plt.subplots(figsize=(3.3, 3.3), constrained_layout=True, dpi=dpi)
    legend_handles = []
    order_array = np.array(order_labels)
    for label in ["forward", "reversed", "scrambled"]:
        mask = order_array == label
        if not mask.any():
            continue
        scatter = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            color=color_map.get(label, "#888888"),
            alpha=0.7,
            label=label.capitalize(),
            edgecolors="none",
        )
        legend_handles.append(scatter)
    legend = ax.legend(
        handles=legend_handles,
        loc="best",
        frameon=True,
        prop={"size": 8},
    )
    legend.set_title("Frame order")
    ax.set_xlabel(r"$Z_\mathrm{1}$", fontsize=10)
    ax.set_ylabel(r"$Z_\mathrm{2}$", fontsize=10)
    ax.tick_params(labelsize=8)
    ensure_parent(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved order-colored UMAP to {path}", flush=True)


def _plot_k(
    embedding: np.ndarray,
    k_values: np.ndarray,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 3.3), constrained_layout=True, dpi=dpi)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=k_values,
        cmap="plasma",
        s=10,
        edgecolors="none",
    )
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$Z_\mathrm{1}$", fontsize=10)
    ax.set_ylabel(r"$Z_\mathrm{2}$", fontsize=10)
    ax.tick_params(labelsize=8)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("K", fontsize=10, rotation=0)
    cbar.ax.tick_params(labelsize=8)
    ensure_parent(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved K-colored UMAP to {path}", flush=True)


def main() -> None:
    args = parse_args()

    def _prepare_arrays(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
        latent_cols = _latent_columns(df)
        latents = df[latent_cols].astype(np.float32).to_numpy()
        order_labels = df["order_label"].fillna("forward").astype(str).tolist()
        if "K_value" not in df.columns:
            raise RuntimeError("Column 'K_value' is required in the latent CSVs.")
        k_values = df["K_value"].astype(np.float32).to_numpy()
        return latents, order_labels, k_values

    datasets: list[tuple[str, Path, Path, Path]] = []
    if args.include_correct:
        datasets.append(("correct order", args.latent_csv, args.order_output, args.k_output))
    if args.include_scrambled:
        datasets.append(("fully scrambled", args.scrambled_latent_csv, args.scrambled_order_output, args.scrambled_k_output))

    any_plotted = False
    for name, csv_path, order_out, k_out in datasets:
        df = _load_dataset(name, csv_path)
        if df is None:
            continue
        latents, order_labels, k_values = _prepare_arrays(df)
        embedding = _compute_embedding(latents, args.n_neighbors, args.min_dist, args.seed)
        dataset_title = f"{args.title} ({name})"
        _plot_order(embedding, order_labels, order_out, args.dpi)
        _plot_k(embedding, k_values, dataset_title, k_out, args.dpi)
        any_plotted = True

    if not any_plotted:
        raise RuntimeError("No datasets were selected; enable --include_correct and/or --include_scrambled.")


if __name__ == "__main__":
    main()
