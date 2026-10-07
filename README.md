python -m flashrag.retriever.index_builder --retrieval_method e5 --model_path intfloat/e5-base-v2 --corpus_path ../documents/wiki-18_part4.jsonl --save_dir ../index/wiki_e5_flat_part4 --use_fp16 --max_length 256 --batch_size 256 --pooling_method mean --faiss_type Flat

cd FlashRAG
python -m flashrag.retriever.index_builder --retrieval_method e5 --model_path intfloat/e5-base-v2 --corpus_path ../documents/wiki-18_part1.jsonl --save_dir ../index/wiki_e5_ivfsq_part1 --use_fp16 --max_length 256 --batch_size 256 --pooling_method mean --faiss_type IVF16384,SQ8