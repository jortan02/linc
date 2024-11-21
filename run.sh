#!/bin/bash
#SBATCH --requeue
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mail-user=jordan.tan@utah.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./out/%A/%a.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate linc

mkdir /scratch/general/vast/u1283221/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1283221/huggingface_cache"

model_name=""
model_choice=""
dataset_name=""
batch_size=0

# Parse named arguments
while [ $# -gt 0 ]; do
  case $1 in
    --model_name)
      model_name="$2"
      shift 2
      ;;
    --model_choice)
      model_choice="$2"
      shift 2
      ;;
    --dataset_name)
      dataset_name="$2"
      shift 2
      ;;
    --batch_size)
      batch_size="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

python runner.py --model_name $model_name --model_choice $model_choice --dataset_name $dataset_name --batch_size $batch_size