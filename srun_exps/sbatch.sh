#!/bin/bash

#SBATCH --job-name=multinode_job        # Job name
#SBATCH --output=output.txt          # Output file
#SBATCH --error=error.txt            # Error file
#SBATCH --nodes=4                       # Number of nodes
#SBATCH --time=12:00:00                 # Time limit (1 hour)
#SBATCH --partition=all             # Partition name
#SBATCH --ntasks-per-node=1

source /home/mp/maxtext/venv/bin/activate
cd /home/mp/my_maxtext
export SLURM_STEP_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
idx=$(date +%Y-%m-%d-%H-%M)
BASE_OUTPUT_DIRECTORY=gs://focused-llama/mpioro/maxtext_exps

# srun env
srun python3 MaxText/train.py MaxText/configs/mpioro/llama2-7b.yml run_name=runner_pretraining_${idx} base_output_directory=${BASE_OUTPUT_DIRECTORY} neptune_project=pmtest/llm-random 
