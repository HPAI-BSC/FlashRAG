singularity exec -B /gpfs/projects/bsc70 --nv /gpfs/projects/bsc70/hpai/storage/data/heka/singularity/flashrag.sif \
  bash -c "export PATH=/miniconda3/bin:$PATH && source activate base && conda activate conda_env && python3.12 -m flashrag.retriever.index_builder \
  --retrieval_method pubmedbert \
  --model_path /gpfs/scratch/bsc70/hpai/storage/projects/heka/models/embeddings/pubmedbert-base-embeddings \
  --corpus_path indexes/rag_data_formatted.jsonl \
  --save_dir indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --faiss_type Flat"