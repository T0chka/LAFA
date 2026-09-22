# LAFA protein function prediction

This repository adapts the [CAFA6 11th-place solution](https://github.com/T0chka/cafa6_11th_place_solution) for longitudinal evaluation with [LAFA](https://functionbench.net/). The model architecture, component configurations, learning-to-rank ensemble, and postprocessing are unchanged. Only snapshot-dependent data preparation, embedding reuse, and the LAFA execution interface are adapted.

## Snapshot semantics

Each LAFA time point is treated as an independent training/evaluation snapshot.

- Experimental training labels come **only** from the current `train_terms.tsv`. UniProt-GOA never supplements or replaces the training labels.
- The current `goa_uniprot_sprot.gaf.gz` is reparsed for each snapshot and is used for current `NOT` annotations and the same ten non-experimental evidence features used by the CAFA6 solution: `IBA`, `IEA`, `IGC`, `IKR`, `ISA`, `ISM`, `ISO`, `ISS`, `NAS`, and `RCA`.
- Information accretion is computed from the current experimental training labels and current GO graph unless an IA file is supplied explicitly.
- Protein-language-model embeddings are cached by SHA1 sequence hash. A new snapshot computes embeddings only for sequences that are not already present in the shared cache.
- Training and prediction always select embeddings through the current snapshot's train/query indices, so proteins retained only in the shared cache do not enter the current training set.
- All trained predictors, BLAST databases/hits, LTR models, prepared labels, and predictions are snapshot-specific and are rebuilt inside the snapshot work directory.

Use a separate `--work_dir` for every LAFA snapshot and a shared `--embedding_cache` across snapshots.

## Solution architecture

| Component | Input / signal | Main configuration |
|---|---|---|
| `hmlp_esm2` | ESM2 3B embeddings (`esm2_t36_3B_UR50D`, 2560) | Hierarchy-aware MLP; 5 folds; hidden 2048; dropout 0.5; batch 512; 100 epochs; max LR `3e-4`; weight decay 0.25; hierarchy-aware parent loss |
| `mlp_t5_esm1b` | ProtT5 (1024) + ESM1b (1280) embeddings | MLP with residual blocks; 5 folds; hidden 2048; dropout 0.6; batch 512; 200 epochs; max LR `1e-3`; weight decay 0.2; plateau scheduler |
| `pyb_t5` | ProtT5 embeddings (1024) | PyBoost; depth 6; learning rate 0.03; colsample 0.8; 5000 trees; `sketch_method="proj"`; `sketch_outputs=1`; prediction batch size 60,000 |
| `blast_knn` | BLAST similarity to current training sequences | E-value ≤ `1e-3`; self-hits removed; highest-bitscore HSP retained per sequence pair; top 30 neighbors for BPO/MFO and 50 for CCO; normalized bitscore-weighted annotation transfer; 5-fold OOF construction |
| `naive_prior` | Current experimental training labels | Per-aspect term frequency: distinct annotated training proteins for a term divided by annotated training proteins in the aspect |
| `nonexp` | Current UniProt-GOA snapshot | IA-ranked binary component plus the ten fixed non-experimental evidence indicators listed above |

The final XGBoost ranker uses the six component scores, GO term information accretion, and the ten non-experimental evidence indicators. Candidate terms are formed from the union of component predictions, ranked with the same LTR configuration as the CAFA6 solution, propagated through the GO hierarchy, and reduced to the final top 500 predictions per protein.

## LAFA inputs

The runner accepts the standard LAFA files plus the current raw UniProt-GOA GAF:

- `--query_file`: current query/test FASTA;
- `--train_sequences`: current training FASTA;
- `--annot_file`: current `train_terms.tsv`;
- `--graph`: current `go-basic.obo`;
- `--goa_gaf`: current `goa_uniprot_sprot.gaf.gz`;
- `--output_file`: three-column prediction TSV, optionally `.gz`.

The LAFA snapshots provide these inputs through the public dataset described in the [LAFA container guide](https://github.com/FriedbergLab/LAFA_container_guide).

## Run one snapshot

```bash
uv run python -m scripts.lafa \
  --query_file /data/test_sequences.fasta \
  --train_sequences /data/train_sequences.fasta \
  --annot_file /data/train_terms.tsv \
  --graph /data/go-basic.obo \
  --goa_gaf /data/goa_uniprot_sprot.gaf.gz \
  --work_dir artifacts/Dec_2025 \
  --embedding_cache artifacts/cache/embeddings \
  --output_file predictions/Dec_2025.tsv
```

On a later snapshot, use a new work directory and the same embedding cache. Existing sequence hashes are reused and only new or changed sequences are embedded.

An external IA file can be supplied with `--ia_file`; otherwise it is computed from the current training snapshot.

## Pipeline

The LAFA runner executes the same modeling sequence as the CAFA6 solution:

```text
prepare snapshot data
generate/reuse ESM2, ProtT5 and ESM1b embeddings
train HMLP
train MLP
train PyBoost
build BLAST-KNN
build naive prior
build non-experimental component
train LTR
predict final submission
```

The individual stage scripts remain available. When they are run directly, `config.dataset()` reads the corresponding `LAFA_*` environment variables.

## Repository layout

```text
config.py                 LAFA input and artifact configuration
scripts/lafa.py           LAFA snapshot runner
scripts/                  individual pipeline stages
src/data/                 snapshot preparation, GOA processing and IA
src/embeddings/           sequence-hash embedding cache
src/models/               unchanged six base predictors
src/candidates/           candidate union construction
src/ltr/                  unchanged XGBoost learning-to-rank ensemble
src/core/                 ontology, CSR, postprocessing, scoring and I/O
```

Generated data, embeddings, model files, BLAST outputs, caches, and predictions live under `artifacts/` by default and are excluded from Git.

## Local longitudinal scoring

Local longitudinal evaluation uses the same ground-truth construction as FunctionBench/LAFA production. For a released snapshot pair, ground truth is built with the official `democafa` workflow and the propagated annotation files described below.

For a window `t0 -> t1`, FunctionBench first intersects the two released test FASTA files with `democafa.datacollection.compare_fasta`. It then runs `democafa.groundtruth.classify_ground_truth` with the **propagated** annotation files from both releases:

```text
t0/train_terms_propagated.tsv
t1/train_terms_propagated.tsv
common unchanged test FASTA
t0/go-basic.obo
t1/go-basic.obo
```

The classifier produces `groundtruth_NK.tsv`, `groundtruth_LK.tsv`, `groundtruth_PK.tsv`, `groundtruth_PK_known.tsv`, `groundtruth_terms_of_interest.txt`, and `groundtruth_targets.tsv`. These files are the only ground-truth inputs used by the local scorer.

One-time setup of the official helper package:

```bash
mkdir -p ~/tools
git clone --depth 1 https://github.com/anphan0828/democafa_package.git ~/tools/democafa_package
```

Example for `Sep_2025 -> Nov_2025`:

```bash
cd ~/tools/democafa_package
OUT="$HOME/data/lafa/evaluation/Sep_2025_to_Nov_2025/production"
rm -rf "$OUT"
mkdir -p "$OUT"
uv run --with obonet --with requests --with tqdm python -m democafa.datacollection.compare_fasta "$HOME/data/lafa/Sep_2025/test_sequences.fasta" "$HOME/data/lafa/Nov_2025/test_sequences.fasta" "$OUT/diff_test_sequences.fasta"
uv run --with obonet --with requests python -m democafa.groundtruth.classify_ground_truth --annot_known "$HOME/data/lafa/Sep_2025/train_terms_propagated.tsv" --annot2 "$HOME/data/lafa/Nov_2025/train_terms_propagated.tsv" --query_file "$OUT/diff_test_sequences_common.fasta" --graph "$HOME/data/lafa/Sep_2025/go-basic.obo" --graph2 "$HOME/data/lafa/Nov_2025/go-basic.obo" --out_prefix "$OUT/groundtruth.tsv"
```

The `Sep_2025 -> Nov_2025` ground-truth regression check is:

```text
NK: 224 unique proteins   BPO=140 CCO=137 MFO=54
LK: 638 unique proteins   BPO=392 CCO=207 MFO=79
PK: 5156 unique proteins  BPO=3063 CCO=1476 MFO=1050
unique across NK/LK/PK: 5898
LK intersect PK: 120
```

Any mismatch means the ground truth was built with the wrong inputs or a different `democafa` implementation. Do not score until these counts match.

Score all seven local methods after the source snapshot has completed prediction:

```bash
cd ~/projects/lafa
uv run python -m scripts.score_eval --groundtruth-dir "$HOME/data/lafa/evaluation/Sep_2025_to_Nov_2025/production" --source-work "$PWD/artifacts/snapshots/Sep_2025" --source-snapshot Sep_2025 --future-snapshot Nov_2025 --output "$PWD/artifacts/evaluation/Sep_2025_to_Nov_2025"
```

The scorer consumes the official `democafa` NK/LK/PK files directly, restricts scoring to `groundtruth_terms_of_interest.txt`, and uses the source snapshot IA weights. Test predictions already exclude source-known terms during model postprocessing, so the local scorer evaluates the supplied ground-truth rows directly. It writes:

```text
scores_long.tsv
scores.tsv
thresholds.tsv
```

`scores.tsv` contains the nine `NK/LK/PK x BPO/CCO/MFO` `f_micro` cells and their mean for `hmlp_esm2`, `mlp_t5_esm1b`, `pyb_t5`, `blast_knn`, `naive_prior`, `nonexp`, and `ltr`. This fast scorer is for local model comparison and longitudinal dynamics; official FunctionBench results remain the external reference score.


## Retraining a source snapshot

A full retrain is required whenever the **source snapshot changes**.

The same trained source snapshot is reused for every later evaluation window. For example:

```text
Sep_2025 model -> Nov_2025, Dec_2025, Mar_2026
Nov_2025 model -> Dec_2025, Mar_2026
Dec_2025 model -> Mar_2026
```

Do **not** retrain separately for each future snapshot. Future-snapshot data are not used during training.

### 0. Download the source snapshot

Before retraining a new source snapshot, download that LAFA release locally if it is not already present under `~/data/lafa/`.

For example, to download the `Dec_2025` release only:

```bash
cd ~/projects/lafa
uvx --from huggingface_hub hf download anphan0828/lafa \
  --repo-type dataset \
  --include "Dec_2025/*" \
  --local-dir "$HOME/data/lafa"
ls -lh "$HOME/data/lafa/Dec_2025"
```

This downloads only the requested release rather than the full LAFA dataset. The resulting snapshot directory is:

```text
~/data/lafa/Dec_2025/
```

Use that directory as the source data directory for the retraining steps below.

### 1. Select the source snapshot

Run from the LAFA repository:

```bash
cd ~/projects/lafa
SOURCE=Sep_2025
DATA="$HOME/data/lafa/$SOURCE"
```

Set the snapshot inputs and shared embedding cache, check that all required source files exist, the resolved configuration:

```bash
export LAFA_TRAIN_SEQUENCES="$DATA/train_sequences.fasta"
export LAFA_QUERY_FILE="$DATA/test_sequences.fasta"
export LAFA_TRAIN_TERMS="$DATA/train_terms.tsv"
export LAFA_GRAPH="$DATA/go-basic.obo"
export LAFA_GOA_GAF="$DATA/goa_uniprot_sprot.gaf.gz"
export LAFA_IA_FILE="$DATA/IA.tsv"
export LAFA_WORK_DIR="$PWD/artifacts/snapshots/$SOURCE"
export LAFA_EMBEDDING_CACHE="/mnt/models/protein_embeddings"

for f in "$LAFA_TRAIN_SEQUENCES" "$LAFA_QUERY_FILE" "$LAFA_TRAIN_TERMS" "$LAFA_GRAPH" "$LAFA_GOA_GAF" "$LAFA_IA_FILE"; do test -s "$f" || { echo "MISSING: $f"; exit 1; }; done

printf 'SOURCE=%s\nTRAIN=%s\nQUERY=%s\nTERMS=%s\nGRAPH=%s\nGAF=%s\nIA=%s\nWORK=%s\nEMBED=%s\n' "$SOURCE" "$LAFA_TRAIN_SEQUENCES" "$LAFA_QUERY_FILE" "$LAFA_TRAIN_TERMS" "$LAFA_GRAPH" "$LAFA_GOA_GAF" "$LAFA_IA_FILE" "$LAFA_WORK_DIR" "$LAFA_EMBEDDING_CACHE"
```

### 2. Start a clean source-snapshot retrain, prepare data

For a true retrain from scratch, remove only the source-specific work directory:

```bash
rm -rf "$LAFA_WORK_DIR"
uv run python -m scripts.prepare_data
uv run python -m scripts.make_embeddings

```
### 3. Train the neural and boosting components

The stage scripts train source-snapshot models and write OOF predictions only. They do not run test inference.

```bash
uv run python -m scripts.build_hmlp
uv run python -m scripts.build_mlp
uv run python -m scripts.build_pyboost
```

### 4. Rebuild BLAST-KNN OOF

BLAST train-vs-train hits are snapshot-dependent. For a later source snapshot with an earlier trained snapshot available, use the incremental builder:

```bash
command -v makeblastdb
command -v blastp
uv run python -m scripts.build_blast_incremental
```

For the first source snapshot, when no earlier train-vs-train BLAST hits exist, use:

```bash
uv run python -m scripts.build_blast
```

Both commands build the BLAST OOF component only. Test-query BLAST is deferred to prediction.

### 5. Rebuild the non-learned OOF components

```bash
uv run python -m scripts.build_naive
uv run python -m scripts.build_nonexp
```

At this point all six member components have source-specific models or state plus OOF predictions under:

```text
artifacts/snapshots/<SOURCE>/predictors/
```

### 6. Train the LTR ensemble

Train LTR only after all six OOF member predictions exist:

```bash
uv run python -m scripts.train_ltr
```

No test inference is required to train the source snapshot.

### 7. Predict an evaluation target set

For a released retrospective window, run every component only on the proteins in that window's `groundtruth_targets.tsv`. Keep predictions outside the source-training directory so multiple future windows can coexist.

Example for `Dec_2025 -> Mar_2026`:

```bash
TARGETS="$HOME/tools/CAFA_forever/data/releases/Dec_2025_Mar_2026/groundtruth_targets.tsv"
PREDICTION_WORK="$PWD/artifacts/predictions/Dec_2025_to_Mar_2026"

uv run python -m scripts.predict --targets "$TARGETS" --prediction-work "$PREDICTION_WORK"
```

This runs HMLP, MLP, PyBoost, BLAST-KNN, naive prior, non-experimental features, and LTR only for the requested EntryIDs. The trained source models and OOF files are reused unchanged.

To run production-style inference for the complete current query FASTA, omit `--targets` and `--prediction-work`:

```bash
uv run python -m scripts.predict
```

The full-query mode writes the traditional component submit files under the source work directory and `final/submission.tsv`.

### 8. Score a target-set prediction

When predictions were written to a separate prediction work directory, pass it explicitly to the scorer:

```bash
uv run python -m scripts.score_eval \
  --groundtruth-dir "$HOME/tools/CAFA_forever/data/releases/Dec_2025_Mar_2026" \
  --source-work "$PWD/artifacts/snapshots/Dec_2025" \
  --prediction-work "$PWD/artifacts/predictions/Dec_2025_to_Mar_2026" \
  --source-snapshot Dec_2025 \
  --future-snapshot Mar_2026 \
  --output "$PWD/artifacts/evaluation/Dec_2025_to_Mar_2026"
```

`--source-work` supplies the frozen source ontology, IA, and training artifacts. `--prediction-work` supplies only the predictions for the selected evaluation targets.

### Retraining schedule

With the currently available source snapshots:

```text
SOURCE=Sep_2025
    train once
    reuse for all Sep_2025 -> future evaluations

SOURCE=Nov_2025
    full retrain
    reuse for all Nov_2025 -> future evaluations

SOURCE=Dec_2025
    full retrain
    reuse for all Dec_2025 -> future evaluations
```

Changing `FUTURE` never triggers retraining. Changing `SOURCE` always does.

## Container

Public Docker image: [tochka897/lafa:v1](https://hub.docker.com/r/tochka897/lafa)

Immutable image: `tochka897/lafa@sha256:2331e556d07e331f18efc92ba23dbd9a0a7928e99f1779d338943c21643d0e13`

Precomputed protein embeddings: [AnDolgorukova/lafa-protein-embeddings](https://huggingface.co/datasets/AnDolgorukova/lafa-protein-embeddings).

Runtime mounts, pretrained protein-language-model preparation, and the complete container command are documented in `CONTAINER.md`.
