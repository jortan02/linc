#!/bin/bash
#SBATCH --requeue
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mail-user=jordan.tan@utah.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./out/%A/%a.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate linc

mkdir /scratch/general/vast/u1283221/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1283221/huggingface_cache"

python runner.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --model_choice "naive" --dataset_name "yale-nlp/FOLIO" --batch_size 2