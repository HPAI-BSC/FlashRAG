#!/bin/bash
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --output=slurm_output/out.txt
#SBATCH --error=slurm_output/err.txt
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

echo "START TIME: $(date)"

module purge
module load singularity

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real/libcuda.so

# singularity exec -B /gpfs/projects/bsc70 --nv /gpfs/projects/bsc70/hpai/storage/data/heka/singularity/flashrag.sif \
#   bash -c "source activate base && conda activate conda_env && python3.12 -m flashrag.retriever.index_builder \
#   --retrieval_method pubmedbert \
#   --model_path NeuML/pubmedbert-base-embeddings \
#   --corpus_path /mnt/c/Users/Jordi/Documents/bsc/repos/flashrag/indexes/rag_data_formatted.jsonl \
#   --save_dir /mnt/c/Users/Jordi/Documents/bsc/repos/flashrag/indexes \
#   --use_fp16 \
#   --sentence_transformer \
#   --max_length 512 \
#   --batch_size 256 \
#   --pooling_method pooler \
#   --faiss_type Flat"

singularity exec -B /gpfs/projects/bsc70 --nv /gpfs/projects/bsc70/hpai/storage/data/heka/singularity/flashrag.sif \
  bash -c "source activate base && conda activate conda_env && python3.12 scripts/run_benchmark.py configs/mriexperts_oscar.yaml"