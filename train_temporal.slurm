#!/bin/bash
#SBATCH --job-name=temporal_cpc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/train_temporal_cpc_%j.out

module purge
source ~/.bashrc
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-matthew}"

WORKDIR="."
mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

python train_temporal_cpc.py \
  --latent_csv "${WORKDIR}/spatial_latent_vectors.csv" \
  --output_dir "${WORKDIR}" \
  --batch_size 16 \
  --lr 1e-3 \
  --max_epochs 400 \
  --patience 40 \
  --segment_length 64 \
  --segment_stride 8 \
  --cpc_weight 1.0 \
  --recon_weight 1.0 \
  --include_reversed_trajectories \
  --include_scrambled_trajectories \
  --scrambled_seed 42 \
  --scramble_order
