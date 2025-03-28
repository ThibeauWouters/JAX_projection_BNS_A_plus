#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G
#SBATCH --output="./outdir/log.out"
#SBATCH --job-name="inference"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/jose

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
python inference.py \
    --eos HQC18 \
    --id 1 2 3 4 5 6 7 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --outdir ./outdir/ \
    --sample-radio True \
    --local-sampler-name MALA \
    --n-loop-training 5 \
    --n-loop-production 5 \
    # --sample-chiEFT True \
    
echo "DONE"