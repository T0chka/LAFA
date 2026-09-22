import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.core.debug import print_separator

from src.data.dataset import DatasetSpec


HIT_COLUMNS = ["qseqid", "sseqid", "bitscore", "evalue"]


def _write_unique_fasta(index_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(index_path, columns=["seq_key", "sequence"])
    df = df.drop_duplicates("seq_key", keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for seq_key, sequence in df.itertuples(index=False):
            f.write(f">{seq_key}\n{sequence}\n")


def _write_index_fasta(index_df: pd.DataFrame, out_path: Path) -> None:
    df = index_df.loc[:, ["seq_key", "sequence"]].drop_duplicates("seq_key", keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for seq_key, sequence in df.itertuples(index=False):
            f.write(f">{seq_key}\n{sequence}\n")


def _db_exists(prefix: Path) -> bool:
    return prefix.with_suffix(".pin").exists() or prefix.with_suffix(".pdb").exists()


def _report_blast_stderr(stderr: str, log_prefix: str) -> None:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    o_warnings = [line for line in lines if "O characters replaced by X" in line]
    if o_warnings:
        print(
            f"[{log_prefix}] warning: non-standard residue O replaced by X in "
            f"{len(o_warnings):,} test sequence(s)"
        )
    for line in lines:
        if line not in o_warnings:
            print(f"[{log_prefix}] warning: {line}")


def _make_db(fasta: Path, prefix: Path) -> None:
    subprocess.run(
        ["makeblastdb", "-in", str(fasta), "-dbtype", "prot", "-parse_seqids", "-out", str(prefix)],
        check=True, capture_output=True, text=True,
    )


def _run_blast(
    query: Path,
    db: Path,
    out_path: Path,
    threads: int,
    log_prefix: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "blastp",
            "-query", str(query),
            "-db", str(db),
            "-max_target_seqs", "500",
            "-evalue", "1e-3",
            "-num_threads", str(threads),
            "-outfmt", "6 qseqid sseqid bitscore evalue",
            "-out", str(out_path),
        ],
        check=True, capture_output=True, text=True,
    )
    _report_blast_stderr(result.stderr, log_prefix)


def _to_parquet(tsv_path: Path, parquet_path: Path) -> int:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=HIT_COLUMNS,
        dtype={"qseqid": "string", "sseqid": "string"},
    )
    df["bitscore"] = pd.to_numeric(df["bitscore"], errors="raise")
    df["evalue"] = pd.to_numeric(df["evalue"], errors="raise")
    df.to_parquet(parquet_path, index=False)
    n_rows = len(df)
    tsv_path.unlink()
    return n_rows


def ensure_blast_train_hits(
    dataset: DatasetSpec,
    threads: int | None = None,
    log_prefix: str = "build_blast",
) -> Path:
    root = dataset.prepared_dir.parent / "blast"
    train_hits = root / "hits_train_vs_train.parquet"
    if train_hits.exists():
        return train_hits

    for executable in ("makeblastdb", "blastp"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required on PATH")

    db_dir = root / "db"
    train_fasta = db_dir / "train_unique.fasta"
    db_prefix = db_dir / "train"
    db_dir.mkdir(parents=True, exist_ok=True)

    train_index = pd.read_parquet(dataset.train_index, columns=["seq_key"])
    print_separator(log_prefix, "Full train-vs-train BLAST search")
    n_seq = train_index["seq_key"].nunique()
    print(f"[{log_prefix}] full train-vs-train search | train sequences={n_seq:,}")
    if not train_fasta.exists():
        _write_unique_fasta(dataset.train_index, train_fasta)
        print(f"[{log_prefix}] wrote train FASTA: {train_fasta}")
    if not _db_exists(db_prefix):
        _make_db(train_fasta, db_prefix)
        print(f"[{log_prefix}] built train BLAST database: {db_prefix}")
    else:
        print(f"[{log_prefix}] using existing train BLAST database: {db_prefix}")

    n_threads = int(threads or min(16, os.cpu_count() or 1))
    train_tsv = root / "hits_train_vs_train.tsv"
    _run_blast(train_fasta, db_prefix, train_tsv, n_threads, log_prefix)
    n_hits = _to_parquet(train_tsv, train_hits)
    print(f"[{log_prefix}] train-vs-train | hit rows={n_hits:,}")
    print(f"[{log_prefix}] wrote train BLAST hits: {train_hits}")
    return train_hits


def _source_db(dataset: DatasetSpec, log_prefix: str = "predict") -> Path:
    root = dataset.prepared_dir.parent
    for prefix in (root / "blast_incremental/db/train", root / "blast/db/train"):
        if _db_exists(prefix):
            return prefix
    train_fasta = root / "blast_prediction_db/train_unique.fasta"
    db_prefix = root / "blast_prediction_db/train"
    if not train_fasta.exists():
        _write_unique_fasta(dataset.train_index, train_fasta)
    if not _db_exists(db_prefix):
        _make_db(train_fasta, db_prefix)
        print(f"[{log_prefix}] built prediction BLAST database: {db_prefix}")
    return db_prefix


def build_blast_query_hits(
    dataset: DatasetSpec,
    index_df: pd.DataFrame,
    out_dir: str | Path,
    threads: int | None = None,
    log_prefix: str = "predict",
) -> Path:
    for executable in ("makeblastdb", "blastp"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required on PATH")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = out_dir / "hits_query.parquet"
    if hits.exists():
        return hits
    query_fasta = out_dir / "query_unique.fasta"
    query_tsv = out_dir / "hits_query.tsv"
    n_proteins = index_df["EntryID"].nunique()
    n_seq = index_df["seq_key"].nunique()
    print_separator(log_prefix, "Test-vs-train BLAST search")
    print(
        f"[{log_prefix}] test-vs-train BLAST search | proteins={n_proteins:,} | "
        f"unique_sequences={n_seq:,}"
    )
    _write_index_fasta(index_df, query_fasta)
    print(f"[{log_prefix}] wrote test FASTA: {query_fasta}")
    n_threads = int(threads or min(16, os.cpu_count() or 1))
    _run_blast(
        query_fasta, _source_db(dataset, log_prefix), query_tsv, n_threads, log_prefix
    )
    n_hits = _to_parquet(query_tsv, hits)
    print(f"[{log_prefix}] test BLAST | hit rows={n_hits:,}")
    print(f"[{log_prefix}] wrote test BLAST hits: {hits}")
    return hits


def ensure_blast_hits(
    dataset: DatasetSpec,
    threads: int | None = None,
    log_prefix: str = "build_blast",
) -> tuple[Path, Path]:
    root = dataset.prepared_dir.parent / "blast"
    train_hits = ensure_blast_train_hits(
        dataset, threads=threads, log_prefix=log_prefix
    )
    test_hits = root / "hits_query.parquet"
    if test_hits.exists():
        return train_hits, test_hits
    test_index = pd.read_parquet(dataset.test_index)
    test_hits = build_blast_query_hits(
        dataset, test_index, root, threads=threads, log_prefix=log_prefix
    )
    return train_hits, test_hits
