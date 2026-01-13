#!/bin/bash
#SBATCH --job-name=spatial_ae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/train_spatial_%j.out

module purge
source ~/.bashrc
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-matthew}"

WORKDIR="."
mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

python train_spatial.py \
  --replicate_data_csv "stuart_landau_trajectories_with_replicates.csv" \
  --use_replicate_dataset \
  --output_dir "${WORKDIR}" \
  --batch_size 4 \
  --lr 1e-3 \
  --max_epochs 250 \
  --patience 10 \
  # NOTE: latent_dim must be 16.
  # Setting latent_dim to 8 is too low: the bottleneck cannot capture required spatial features,
  # which causes the downstream temporal step to fail (instability/poor learning or divergence).
  # Do not reduce latent_dim below 16 if you expect the temporal processing stage to work reliably.
  --latent_dim 16
