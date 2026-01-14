# Stuart-Landau Autoencoded Time Series Mapping (AuTSM) Demonstration

This repository trains a transformer autoencoder that treats oscillators as tokens within each frame, producing a per-frame CLS latent and reconstructing the same frame. It then fits a temporal CPC model over those latents. All paths are relative, so run commands from the repo root.

## Quick start

1. Install Git LFS and pull the dataset:
   ```bash
   git lfs install
   git lfs pull
   ```
2. Submit the full pipeline (spatial + temporal + UMAP):
   ```bash
   sbatch full_train.slurm
   ```

## Scripts

- `train_spatial.py`: trains a per-frame transformer autoencoder over oscillators (CLS latent + frame reconstruction), writes `spatial_latent_vectors.csv`, checkpoints, and training logs. Optional `--test_recon_quality` exports reconstruction videos.
- `train_temporal_cpc.py`: trains the temporal CPC model on the spatial latents, writes spatiotemporal latent CSVs and checkpoints.
- `plot_umap_coloredby_order.py`: projects spatiotemporal latents with UMAP and saves order-colored/coupling-strength-colored PNGs.

## Slurm entry points

- `full_train.slurm`: runs the full pipeline (spatial training, temporal CPC, UMAP plots). **Start here.**
- `train_spatial.slurm`: spatial autoencoder only.
- `train_temporal.slurm`: temporal CPC only (expects `spatial_latent_vectors.csv`).
- `plot_umap_coloredby_order.slurm`: UMAP plots only (expects spatiotemporal latent CSVs).

## Data

- `stuart_landau_trajectories_with_replicates.csv` is required and tracked with Git LFS. Make sure LFS is installed and the file is present after cloning.

## Outputs
- spatiotemporal_latents_coloredby_k.png: projection of trajectory embeddings to 2D, colored by coupling strength. This plot reveals three distinct manifolds, each parametrized by coupling strength (K), revealing the expected outcome that coupling strength determines similarity.
- spatiotemporal_latents_coloredby_order.png: the same projection as above, colored by integration order (forward/reversed/scrambled). This plot reveals three distinct manifolds, each corresponding to a different order, recovering the expected dynamical regimes.

## Conda environment requirements

Install these packages in your conda environment (CUDA-enabled PyTorch recommended if available):

- Python 3.9+
- `numpy`
- `pandas`
- `matplotlib`
- `pytorch`
- `umap-learn` (required for UMAP plotting)
- `ffmpeg` (required only for reconstruction videos in `train_spatial.py --test_recon_quality`)

## Update your conda env name

The Slurm scripts use:

```bash
conda activate "${CONDA_ENV:-matthew}"
```

Replace `matthew` with your environment name, or export `CONDA_ENV` before submitting jobs.
