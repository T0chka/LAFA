# LAFA container runtime

Public image: [tochka897/lafa:v1](https://hub.docker.com/r/tochka897/lafa)

Immutable image: `tochka897/lafa@sha256:2331e556d07e331f18efc92ba23dbd9a0a7928e99f1779d338943c21643d0e13`

The Docker image contains the LAFA code, Python/CUDA dependencies, and BLAST. Large reusable data are mounted separately rather than included in the image.

## Runtime mounts

- `/data` — LAFA snapshot input, read-only.
- `/embeddings` — protein embedding cache, read-write.
- `/models` — pretrained protein-language-model files, read-only.
- `/work` — run workspace and final submission, read-write.

The embedding cache is keyed by protein sequence. Existing embeddings are reused; missing embeddings are computed and appended to `/embeddings`. If the mounted cache is preserved between evaluation windows, newly computed embeddings are reused automatically on later runs. If it is not preserved, each run starts from the published base cache and recomputes only sequences absent from that base cache.

## Published protein embeddings

Public dataset:

[AnDolgorukova/lafa-protein-embeddings](https://huggingface.co/datasets/AnDolgorukova/lafa-protein-embeddings)

Download it without changing its directory layout:

```bash
uvx --from huggingface_hub hf download \
  AnDolgorukova/lafa-protein-embeddings \
  --repo-type dataset \
  --local-dir /path/to/lafa_embeddings
```

Mount `/path/to/lafa_embeddings` at `/embeddings` read-write.

## Pretrained protein-language models

The runtime expects this host-side layout:

```text
/path/to/lafa_pretrained/
├── torch/
│   └── hub/
│       └── checkpoints/
│           ├── esm2_t36_3B_UR50D.pt
│           └── esm1b_t33_650M_UR50S.pt
└── huggingface/
    └── hub/
        └── models--Rostlab--prot_t5_xl_uniref50/
            └── ... Hugging Face cache files ...
```

Prepare the two ESM checkpoints from the upstream FAIR ESM release:

```bash
PRETRAINED=/path/to/lafa_pretrained
mkdir -p "$PRETRAINED/torch/hub/checkpoints" "$PRETRAINED/huggingface/hub"

wget -c \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt \
  -O "$PRETRAINED/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"

wget -c \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt \
  -O "$PRETRAINED/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt"
```

Prepare only the ProtT5 files used by the pipeline, pinned to the tested Hugging Face revision:

```bash
PRETRAINED=/path/to/lafa_pretrained
HF_HOME="$PRETRAINED/huggingface" \
uvx --from huggingface_hub hf download \
  Rostlab/prot_t5_xl_uniref50 \
  config.json \
  pytorch_model.bin \
  spiece.model \
  special_tokens_map.json \
  tokenizer_config.json \
  --revision 973be27c52ee6474de9c945952a8008aeb2a1a73
```

Mount `/path/to/lafa_pretrained` at `/models` read-only.

## Build

```bash
docker build --progress=plain -t lafa:dev .
```

## Run

```bash
docker run --rm --gpus all \
  -v /path/to/lafa_snapshot:/data:ro \
  -v /path/to/lafa_embeddings:/embeddings \
  -v /path/to/lafa_pretrained:/models:ro \
  -v /path/to/lafa_work:/work \
  tochka897/lafa:v1 \
  --query_file /data/test_sequences.fasta \
  --train_sequences /data/train_sequences.fasta \
  --annot_file /data/train_terms.tsv \
  --graph /data/go-basic.obo \
  --goa_gaf /data/goa_uniprot_sprot.gaf.gz \
  --work_dir /work/run \
  --embedding_cache /embeddings \
  --output_file /work/submission.tsv
```


The container runs with Hugging Face and Transformers offline mode enabled. All pretrained model files therefore need to be present in `/models` before execution.
