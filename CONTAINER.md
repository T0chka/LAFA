# LAFA container runtime layout

The Docker image contains only the LAFA code, Python/CUDA dependencies, and BLAST.
Training data, pretrained protein-language-model caches, protein embeddings, and run outputs are mounted from the host.

## Mounts

- `/data` — LAFA snapshot input, read-only.
- `/embeddings` — protein embedding cache, writable.
- `/root/.cache/huggingface` — pretrained Hugging Face model cache, read-only.
- `/root/.cache/torch` — pretrained ESM/Torch cache, read-only.
- `/work` — writable working directory and final submission.

The published embedding dataset is mounted at `/embeddings` without `:ro`. Existing sequence embeddings are reused by `seq_key`; embeddings missing for a new snapshot are computed and written into the same cache. If the mounted volume persists between evaluation windows, these additions are automatically reused later. If it does not persist, each run starts from the published base cache and only recomputes sequences absent from that base cache.

## Local build

```bash
cd ~/projects/lafa
docker build --progress=plain -t lafa:dev .
```

## Local smoke test

```bash
rm -rf /tmp/lafa_container_smoke
mkdir -p /tmp/lafa_container_smoke

docker run --rm --gpus all \
  -v /tmp/lafa_dev_snapshot:/data:ro \
  -v /tmp/lafa_container_smoke:/work \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$HOME/.cache/torch:/root/.cache/torch:ro" \
  -v /mnt/models/protein_embeddings:/embeddings \
  lafa:dev \
  --query_file /data/test_sequences.fasta \
  --train_sequences /data/train_sequences.fasta \
  --annot_file /data/train_terms.tsv \
  --graph /data/go-basic.obo \
  --goa_gaf /data/goa_uniprot_sprot.gaf.gz \
  --work_dir /work/run \
  --embedding_cache /embeddings \
  --output_file /work/submission.tsv
```
