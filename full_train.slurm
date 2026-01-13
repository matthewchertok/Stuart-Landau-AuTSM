#!/bin/bash
#SBATCH --job-name=full_autoenc_cpc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/full_train_cpc_%j.out

module purge
source ~/.bashrc
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-matthew}"

WORKDIR="."
DATA_CSV="stuart_landau_trajectories_with_replicates.csv"
LATENT_CSV="${WORKDIR}/spatial_latent_vectors.csv"
mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

echo "[$(date)] Starting spatial autoencoder training" >&2
python train_spatial.py \
  --data_csv "${DATA_CSV}" \
  --use_replicate_dataset \
  --output_dir "${WORKDIR}" \
  --batch_size 16 \
  --lr 1e-3 \
  --max_epochs 250 \
  --patience 10 \
  --latent_dim 16 \
  --test_recon_quality

if [[ ! -f "${LATENT_CSV}" ]]; then
  echo "Spatial training finished but ${LATENT_CSV} is missing; aborting." >&2
  exit 1
fi

echo "[$(date)] Starting temporal CPC training" >&2
python train_temporal_cpc.py \
  --latent_csv "${LATENT_CSV}" \
  --output_dir "${WORKDIR}" \
  --batch_size 16 \
  --lr 1e-3 \
  --max_epochs 400 \
  --patience 40 \
  --segment_length 64 \
  --segment_stride 8 \
  --cpc_weight 1.0 \
  --recon_weight 1.0 \
  --scramble_order \
  --include_scrambled_trajectories \
  --include_reversed_trajectories \
  --scrambled_seed 42

echo "[$(date)] Generating order-colored UMAP" >&2
python plot_umap_coloredby_order.py \
  --latent_csv "${WORKDIR}/spatiotemporal_latent_vectors_correct_order.csv" \
  --title "Spatiotemporal Latents"

echo "[$(date)] Full training pipeline complete" >&2
