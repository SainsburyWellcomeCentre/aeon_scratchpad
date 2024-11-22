#!/bin/bash

#SBATCH --job-name=sleap_optuna_train   # job name
#SBATCH --partition=gpu_lowp            # partition (queue) # gpu_h100 # gpu_l40s # a100
#SBATCH --gres=gpu:1                    # number of gpus per node 
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --mem=128G                      # total memory per node 
#SBATCH --time=06:00:00                 # total run time limit (DD-HH:MM:SS)
#SBATCH --output=slurm_output/%N_%j.out # output file path

mkdir -p slurm_output
source ~/.bashrc
conda activate sleap_133 
python main.py
