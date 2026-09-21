"""
Generate sequence embeddings for the prepared dataset.

Inputs:
- DatasetSpec.train_index and DatasetSpec.test_index, each with columns:
  EntryID, seq_key, length, sequence.
- EmbeddingSpec describing the pretrained protein language model.

Output:
shared embedding cache/<embedding-name>/part-XXXXX.parquet

Each Parquet row contains:
- seq_key: SHA1 sequence key used by the prepared dataset;
- length: sequence length actually represented after truncation;
- embedding: mean-pooled vector stored in the dtype defined by EmbeddingSpec.

Embedding behavior:
- train and test sequences are deduplicated jointly by seq_key;
- sequences are sorted by length before batching;
- sequences are truncated to 1022 residues;
- token batches are limited to approximately 8192 residues/tokens;
- ESM representations are mean-pooled over residue tokens;
- ProtT5 uses Rostlab/prot_t5_xl_uniref50 and the same residue replacement,
  tokenization and pooling logic as the original solution;
- CUDA OOM causes recursive batch splitting;
- output is restartable: seq_keys already present in part-*.parquet are skipped;
- output parts contain at most 4096 rows and use zstd compression.

The ESM1b artifact name used by the original solution was "esm1b_650M".
Its fair-esm checkpoint is esm1b_t33_650M_UR50S.

This module has no dataset-specific paths and no command-line interface.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.data.dataset import DatasetSpec
from src.embeddings.specs import EmbeddingSpec


DEVICE = "cuda"
MAX_TOKENS = 8192
MAX_SEQ_LEN = 1022
ROWS_PER_PARQUET = 4096
PRINT_EVERY_SECONDS = 60    

def _log(msg):
    print(msg, flush=True)

def embedding_dir(dataset: DatasetSpec, spec: EmbeddingSpec) -> Path:
    return dataset.embedding_cache_dir / spec.name


def _token_batches(df: pd.DataFrame, max_tokens: int):
    lengths = df["length"].to_numpy()
    seqs = df["sequence"].tolist()
    keys = df["seq_key"].tolist()
    i = 0
    while i < len(df):
        total = 0
        j = i
        while j < len(df):
            need = int(lengths[j]) + 2
            if j > i and total + need > max_tokens:
                break
            total += need
            j += 1
        yield keys[i:j], seqs[i:j], lengths[i:j], total
        i = j


def existing_seq_keys(out_dir: Path) -> tuple[set[str], list[Path]]:
    tmp_paths = sorted(out_dir.glob("part-*.parquet.tmp"))
    if tmp_paths:
        _log(f"[embed] {out_dir.name}: removing {len(tmp_paths)} incomplete temp shard(s)")
        for path in tmp_paths:
            path.unlink()

    keys: set[str] = set()
    part_paths: list[Path] = []

    for path in sorted(out_dir.glob("part-*.parquet")):
        try:
            table = pq.read_table(path, columns=["seq_key"])
            if table.num_rows == 0:
                raise RuntimeError("empty shard")
        except Exception as exc:
            _log(f"[embed] {out_dir.name}: removing corrupt shard {path.name}: {exc}")
            path.unlink()
            continue

        keys.update(table.column("seq_key").to_pylist())
        part_paths.append(path)

    return keys, part_paths


def _next_part_index(part_paths: list[Path]) -> int:
    if not part_paths:
        return 0
    last = part_paths[-1].stem.split("-")[-1]
    return int(last) + 1 if last.isdigit() else len(part_paths)


def _write_parquet(
    rows: list[tuple[str, int, np.ndarray]],
    out_dir: Path,
    part_idx: int,
    spec: EmbeddingSpec
) -> Path:
    keys, lengths, embeddings = zip(*rows)
    dtype = pa.float16() if spec.storage_dtype == "float16" else pa.float32()
    table = pa.table({
        "seq_key": pa.array(keys),
        "length": pa.array(lengths, type=pa.int32()),
        "embedding": pa.array(
            [x.astype(spec.storage_dtype).tolist() for x in embeddings],
            type=pa.list_(dtype),
        ),
    })

    out_path = out_dir / f"part-{part_idx:05d}.parquet"
    tmp_path = out_dir / f"part-{part_idx:05d}.parquet.tmp"
    tmp_path.unlink(missing_ok=True)

    try:
        pq.write_table(table, tmp_path, compression="zstd")
        pf = pq.ParquetFile(tmp_path)
        if pf.metadata.num_rows != len(rows):
            raise RuntimeError(
                f"Shard row count mismatch: {pf.metadata.num_rows} != {len(rows)}"
            )
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return out_path


def _load_model(spec: EmbeddingSpec, device: str):
    t0 = time.perf_counter()

    if spec.backend == "prott5":
        from transformers import AutoTokenizer, T5EncoderModel

        cache_dir = Path.home() / ".cache/huggingface/hub" / (
            "models--" + spec.model_id.replace("/", "--")
        )
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"ProtT5 cache is missing: {cache_dir}")

        print(f"[embed] {spec.name}: loading local HF model {spec.model_id}", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id, use_fast=False, local_files_only=True
            )
            model = T5EncoderModel.from_pretrained(
                spec.model_id, local_files_only=True
            )
        except OSError as exc:
            raise RuntimeError(
                f"ProtT5 cache is incomplete: {cache_dir}"
            ) from exc

        model = model.to(device).eval()
        print(
            f"[embed] {spec.name}: model ready in "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return model, tokenizer, None

    import esm

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / f"{spec.model_id}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"ESM checkpoint is missing: {checkpoint}"
        )

    size_gib = checkpoint.stat().st_size / 1024**3
    print(
        f"[embed] {spec.name}: loading local checkpoint "
        f"{size_gib:.2f} GiB",
        flush=True,
    )

    model_data = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Regression weights not found.*"
        )
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            spec.model_id, model_data, regression_data=None
        )

    model = model.eval().to(device)
    print(
        f"[embed] {spec.name}: model ready in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return model, alphabet.get_batch_converter(), model.num_layers


def _run_batch(
    keys: list[str],
    seqs: list[str],
    lengths,
    spec: EmbeddingSpec,
    model,
    helper,
    layer,
    device: str,
) -> list[tuple[str, int, np.ndarray]]:
    seqs = [seq[:MAX_SEQ_LEN] for seq in seqs]
    lengths = [min(int(length), MAX_SEQ_LEN) for length in lengths]
    rows: list[tuple[str, int, np.ndarray]] = []

    if spec.backend == "prott5":
        tokenizer = helper
        seqs = [
            seq.translate(str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"}))
            for seq in seqs
        ]
        spaced = [" ".join(list(seq)) for seq in seqs]
        encoded = tokenizer(
            spaced,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN + 2,
            return_tensors="pt",
            return_attention_mask=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden_states = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"]
        for i, key in enumerate(keys):
            valid = mask[i].bool()
            if valid.sum().item() >= 2:
                valid[0] = False
                valid[valid.nonzero()[-1].item()] = False
            emb = hidden_states[i][valid].mean(0).cpu().numpy()
            rows.append((key, int(lengths[i]), emb))
        return rows

    batch = [(key, seq) for key, seq in zip(keys, seqs)]
    _, _, tokens = helper(batch)
    tokens = tokens.to(device)
    reps = model(tokens, repr_layers=[layer])["representations"][layer]
    for i, length in enumerate(lengths):
        emb = reps[i, 1 : length + 1].mean(0).cpu().numpy()
        rows.append((keys[i], int(length), emb))
    return rows


def _run_with_split(
    keys,
    seqs,
    lengths,
    spec: EmbeddingSpec,
    model,
    helper,
    layer,
    device: str,
) -> list[tuple[str, int, np.ndarray]]:
    stack = [(keys, seqs, list(lengths))]
    rows: list[tuple[str, int, np.ndarray]] = []
    while stack:
        keys_i, seqs_i, lengths_i = stack.pop()
        try:
            rows.extend(
                _run_batch(
                    keys_i, seqs_i, lengths_i, spec, model, helper, layer, device
                )
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(keys_i) == 1:
                raise
            mid = len(keys_i) // 2
            stack.append((keys_i[mid:], seqs_i[mid:], lengths_i[mid:]))
            stack.append((keys_i[:mid], seqs_i[:mid], lengths_i[:mid]))
    return rows


def generate_embeddings(dataset: DatasetSpec, spec: EmbeddingSpec, device: str = DEVICE) -> None:
    out_dir = embedding_dir(dataset, spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_prep = time.perf_counter()

    t0 = time.perf_counter()
    train_df = pd.read_parquet(dataset.train_index, columns=["seq_key", "sequence", "length"])
    dt = time.perf_counter() - t0
    if dt >= 1:
        _log(f"[embed] {spec.name}: read train index in {dt:.1f}s")

    t0 = time.perf_counter()
    test_df = pd.read_parquet(dataset.test_index, columns=["seq_key", "sequence", "length"])
    dt = time.perf_counter() - t0
    if dt >= 1:
        _log(f"[embed] {spec.name}: read test index in {dt:.1f}s")

    t0 = time.perf_counter()
    combined = pd.concat([train_df, test_df], ignore_index=True)
    unique_df = (
        combined.drop_duplicates(subset=["seq_key"], keep="first")
        .sort_values("length")
        .reset_index(drop=True)
    )
    del combined
    dt = time.perf_counter() - t0
    if dt >= 1:
        _log(f"[embed] {spec.name}: deduplicated/sorted in {dt:.1f}s")

    total_unique = len(unique_df)

    t0 = time.perf_counter()
    done_keys, part_paths = existing_seq_keys(out_dir)
    dt = time.perf_counter() - t0
    if dt >= 1:
        _log(f"[embed] {spec.name}: scanned {len(part_paths)} cache shards in {dt:.1f}s")

    part = _next_part_index(part_paths)

    t0 = time.perf_counter()
    if done_keys:
        unique_df = unique_df.loc[~unique_df["seq_key"].isin(done_keys)].copy()
    dt = time.perf_counter() - t0
    if dt >= 1:
        _log(f"[embed] {spec.name}: filtered cache hits in {dt:.1f}s")

    remaining = len(unique_df)
    prep_time = time.perf_counter() - t_prep

    _log(
        f"[embed] {spec.name}: required={total_unique:,} cached={len(done_keys):,} "
        f"remaining={remaining:,} | prep={prep_time:.1f}s"
    )

    if remaining == 0:
        _log(f"[embed] {spec.name}: complete, nothing to do")
        return

    model, helper, layer = _load_model(spec, device)

    buffer: list[tuple[str, int, np.ndarray]] = []
    processed = 0
    t0 = time.time()
    last_print = t0

    with torch.inference_mode():
        for keys, seqs, lengths, _ in _token_batches(unique_df, MAX_TOKENS):
            rows = _run_with_split(keys, seqs, lengths, spec, model, helper, layer, device)
            buffer.extend(rows)
            processed += len(keys)

            while len(buffer) >= ROWS_PER_PARQUET:
                _write_parquet(buffer[:ROWS_PER_PARQUET], out_dir, part, spec)
                buffer = buffer[ROWS_PER_PARQUET:]
                part += 1

            now = time.time()
            if now - last_print >= PRINT_EVERY_SECONDS:
                rate = processed / max(now - t0, 1e-9)
                pct = 100.0 * processed / remaining
                eta = (remaining - processed) / rate / 60
                _log(
                    f"[embed] {spec.name}: {processed:,}/{remaining:,} ({pct:.1f}%) | "
                    f"{rate:.1f} seq/s | ETA {eta:.1f} min"
                )
                last_print = now

    if buffer:
        _write_parquet(buffer, out_dir, part, spec)

    elapsed = time.time() - t0
    _log(
        f"[embed] {spec.name}: computed {processed:,} in {elapsed / 60:.1f} min "
        f"({processed / elapsed:.1f} seq/s)"
    )

    required = set(train_df["seq_key"].unique()) | set(test_df["seq_key"].unique())
    embedded, _ = existing_seq_keys(out_dir)
    missing = required - embedded

    if missing:
        raise RuntimeError(f"{len(missing)} required seq_keys are missing embeddings.")

    _log(f"[embed] {spec.name}: verified {len(required):,} required seq_keys")


def embed_sequences(
    sequences: pd.DataFrame,
    spec: EmbeddingSpec,
    device: str = DEVICE,
) -> pd.DataFrame:
    required = {"seq_key", "sequence", "length"}
    missing = required - set(sequences.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    model, helper, layer = _load_model(spec, device)
    rows: list[tuple[str, int, np.ndarray]] = []
    ordered = sequences.loc[:, ["seq_key", "sequence", "length"]].reset_index(drop=True)

    with torch.inference_mode():
        for keys, seqs, lengths, _ in _token_batches(ordered, MAX_TOKENS):
            rows.extend(
                _run_with_split(
                    keys, seqs, lengths, spec, model, helper, layer, device
                )
            )

    return pd.DataFrame(rows, columns=["seq_key", "length", "embedding"])
