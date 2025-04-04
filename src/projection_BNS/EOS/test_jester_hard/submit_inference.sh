#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G
#SBATCH --output="./test_jester_hard/log.out"
#SBATCH --job-name="test"

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
    --eos jester_hard \
    --ifo-network Aplus \
    --id-begin 1 \
    --id-end 30 \
    --outdir ./test_jester_hard/ \
    --local-sampler-name GaussianRandomWalk \
    --eps-mass-matrix 1e-5 \
    --n-loop-training 20 \
    --n-loop-production 20 \
    --n-local-steps 2 \
    --n-global-steps 1000 \
    --n-chains 100 \
    --output-thinning 1 \
    
echo "DONE"