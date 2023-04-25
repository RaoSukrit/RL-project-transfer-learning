#!/bin/bash

#SBATCH --job-name=DDPG-walker-stand
#SBATCH --open-mode=append
#SBATCH --output=/scratch/str8775/conda_env/rl_project/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/str8775/conda_env/rl_project/logs/slurm/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=16G

export ENV=$1

singularity exec --bind /scratch --overlay /scratch/str8775/conda_env/rl_project/rl_project_env${ENV}.ext3:rw /scratch/str8775/conda_env/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "
source /ext3/env_miniforge3.sh
conda activate rl_project
cd /scratch/str8775/conda_env/rl_project/RL-project-transfer-learning
python3 train.py --config=./configs/walker_stand_ddpg_dmcnn.yaml
"
