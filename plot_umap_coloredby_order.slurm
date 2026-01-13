#!/bin/bash
#SBATCH --job-name=order_umap
#SBATCH --output=logs/plot_umap_order_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:02:00

module purge
source ~/.bashrc
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-matthew}"

WORKDIR="."
cd "${WORKDIR}"

STANDARD_CMD=(
  python plot_umap_coloredby_order.py
  --latent_csv "${WORKDIR}/spatiotemporal_latent_vectors_correct_order.csv"
  --scrambled_latent_csv "${WORKDIR}/spatiotemporal_latent_vectors_scrambled.csv"
  --no-include_scrambled
  --title "Spatiotemporal Latents"
)

SCRAMBLED_CMD=(
  python plot_umap_coloredby_order.py
  --latent_csv "${WORKDIR}/spatiotemporal_latent_vectors_correct_order.csv"
  --scrambled_latent_csv "${WORKDIR}/spatiotemporal_latent_vectors_scrambled.csv"
  --no-include_correct
  --title "Spatiotemporal Latents"
)

srun --exclusive -N1 -n1 -c1 "${STANDARD_CMD[@]}" &
srun --exclusive -N1 -n1 -c1 "${SCRAMBLED_CMD[@]}" &
wait
