"""Build the BLAST-KNN component using incremental BLAST hit reuse."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from config import dataset
from src.core.io import load_index_df, load_terms_df
from src.models.blast_knn import BlastKNNConfig, _folds, build_blast_oof_component


HIT_COLUMNS = ["qseqid", "sseqid", "bitscore", "evalue"]
MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _snapshot_key(name: str) -> tuple[int, int] | None:
    """Parse a snapshot directory name into a sortable key."""
    match = re.fullmatch(r"([A-Z][a-z]{2})_(\d{4})", name)
    if match is None or match.group(1) not in MONTHS:
        return None
    return int(match.group(2)), MONTHS[match.group(1)]


def _previous_snapshot(current: Path) -> Path:
    """Find the latest earlier snapshot with reusable train-vs-train hits."""
    current_key = _snapshot_key(current.name)
    if current_key is None:
        raise RuntimeError(f"Cannot parse snapshot name: {current.name}")

    candidates = []
    for path in current.parent.iterdir():
        key = _snapshot_key(path.name)
        if not path.is_dir() or key is None or key >= current_key:
            continue
        full_hits = path / "blast/hits_train_vs_train.parquet"
        incr_hits = path / "blast_incremental/hits_train_vs_train.parquet"
        if (path / "prepared/train_index.parquet").exists() and (
            full_hits.exists() or incr_hits.exists()
        ):
            candidates.append((key, path))

    if not candidates:
        raise RuntimeError(
            f"No earlier snapshot with train BLAST hits found under {current.parent}"
        )

    return max(candidates, key=lambda item: item[0])[1]


def _previous_hits(previous: Path, filename: str) -> Path | None:
    """Return the preferred reusable BLAST hit file for a snapshot."""
    full_hits = previous / "blast" / filename
    if full_hits.exists():
        return full_hits
    incr_hits = previous / "blast_incremental" / filename
    if incr_hits.exists():
        return incr_hits
    return None


def _unique_sequences(index_path: Path) -> pd.DataFrame:
    """Load one row per unique sequence key."""
    df = pd.read_parquet(index_path, columns=["seq_key", "length", "sequence"])
    return df.drop_duplicates("seq_key", keep="first").reset_index(drop=True)


def _write_fasta(df: pd.DataFrame, path: Path) -> None:
    """Write sequence-key FASTA from a prepared index subset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        rows = df.loc[:, ["seq_key", "sequence"]].itertuples(index=False)
        for seq_key, sequence in rows:
            handle.write(f">{seq_key}\n{sequence}\n")


def _make_db(fasta: Path, prefix: Path) -> bool:
    """Build a BLAST protein database when it does not already exist."""
    if prefix.with_suffix(".pin").exists() or prefix.with_suffix(".pdb").exists():
        return False
    prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "makeblastdb", "-in", str(fasta), "-dbtype", "prot",
            "-parse_seqids", "-out", str(prefix),
        ],
        check=True, capture_output=True, text=True,
    )
    return True


def _report_blast_stderr(stderr: str) -> None:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    o_warnings = [line for line in lines if "O characters replaced by X" in line]
    if o_warnings:
        print(
            f"[blast] warning: non-standard residue O replaced by X in "
            f"{len(o_warnings):,} query sequence(s)"
        )
    for line in lines:
        if line not in o_warnings:
            print(f"[blast] warning: {line}")


def _run_blast(
    query: Path,
    db: Path,
    out_path: Path,
    threads: int,
    dbsize: int | None = None,
) -> None:
    """Run BLASTP with the same search settings as the full pipeline."""
    args = [
        "blastp", "-query", str(query), "-db", str(db),
        "-max_target_seqs", "500", "-evalue", "1e-3",
        "-num_threads", str(threads),
        "-outfmt", "6 qseqid sseqid bitscore evalue",
        "-out", str(out_path),
    ]
    if dbsize is not None:
        args.extend(["-dbsize", str(int(dbsize))])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    _report_blast_stderr(result.stderr)


def _read_hits(path: Path) -> pd.DataFrame:
    """Read BLAST TSV or Parquet into the canonical four columns."""
    if path.suffix == ".parquet":
        hits = pd.read_parquet(path, columns=HIT_COLUMNS)
    else:
        hits = pd.read_csv(
            path, sep="\t", header=None, names=HIT_COLUMNS,
            dtype={"qseqid": "string", "sseqid": "string"},
        )
    hits["qseqid"] = hits["qseqid"].astype("string")
    hits["sseqid"] = hits["sseqid"].astype("string")
    hits["bitscore"] = pd.to_numeric(hits["bitscore"], errors="raise")
    hits["evalue"] = pd.to_numeric(hits["evalue"], errors="raise")
    return hits


def _collapse_top500(hits: pd.DataFrame, evalue_max: float) -> pd.DataFrame:
    """Collapse HSPs and retain the top 500 subjects per query."""
    if hits.empty:
        return hits.loc[:, HIT_COLUMNS].copy()
    hits = hits.loc[
        hits["evalue"] <= np.float64(evalue_max), HIT_COLUMNS
    ].copy()
    hits = hits.sort_values(
        ["qseqid", "sseqid", "bitscore"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    hits = hits.drop_duplicates(["qseqid", "sseqid"], keep="first")
    hits = hits.sort_values(
        ["qseqid", "bitscore"], ascending=[True, False], kind="mergesort"
    )
    hits = hits.groupby("qseqid", sort=False, as_index=False).head(500)
    return hits.reset_index(drop=True)


def _donor_sets(
    train_index: pd.DataFrame,
    train_terms_path: Path,
) -> dict[str, set[str]]:
    """Build current sequence-key donor sets for each ontology aspect."""
    terms = load_terms_df(train_terms_path)
    entry_to_seq = dict(
        zip(
            train_index["EntryID"].astype(str),
            train_index["seq_key"].astype(str),
        )
    )
    terms["seq_key"] = terms["EntryID"].astype(str).map(entry_to_seq)
    terms = terms.dropna(subset=["seq_key"])
    return {
        aspect: set(
            terms.loc[terms["aspect"] == aspect, "seq_key"].astype(str)
        )
        for aspect in ("BPO", "CCO", "MFO")
    }


def _insufficient_queries(
    hits: pd.DataFrame,
    donor_sets: dict[str, set[str]],
    config: BlastKNNConfig,
    query_ids: set[str] | None = None,
) -> set[str]:
    """Find queries lacking enough retained donors for at least one aspect."""
    base = hits.loc[hits["qseqid"] != hits["sseqid"], ["qseqid", "sseqid"]]
    if query_ids is None:
        query_ids = set(hits["qseqid"].astype(str).unique())
    insufficient = set()
    for aspect, k in config.k_neighbors.items():
        block = base.loc[base["sseqid"].isin(donor_sets[aspect])]
        counts = block.groupby("qseqid", sort=False)["sseqid"].nunique()
        enough = set(counts.loc[counts >= int(k)].index.astype(str))
        insufficient.update(query_ids - enough)
    return insufficient


def _insufficient_oof_queries(
    hits: pd.DataFrame,
    seq_keys: np.ndarray,
    donor_sets: dict[str, set[str]],
    config: BlastKNNConfig,
    query_ids: set[str],
) -> set[str]:
    """Find candidate queries lacking enough donors inside their OOF fold."""
    if not query_ids:
        return set()
    insufficient = set()
    for train_idx, valid_idx in _folds(
        len(seq_keys), config.n_folds, config.seed
    ):
        valid = set(seq_keys[valid_idx].astype(str)) & query_ids
        if not valid:
            continue
        train_keys = set(seq_keys[train_idx].astype(str))
        fold_hits = hits.loc[
            hits["qseqid"].isin(valid) & hits["sseqid"].isin(train_keys)
        ]
        fold_donors = {
            aspect: donors & train_keys
            for aspect, donors in donor_sets.items()
        }
        insufficient.update(
            _insufficient_queries(
                fold_hits, fold_donors, config, query_ids=valid
            )
        )
    return insufficient


def _replace_queries(
    base: pd.DataFrame,
    replacement: pd.DataFrame,
    query_ids: set[str],
) -> pd.DataFrame:
    """Replace selected query rows with full-search results."""
    if not query_ids:
        return base
    kept = base.loc[~base["qseqid"].isin(query_ids)]
    return pd.concat([kept, replacement], ignore_index=True)


def _full_train_hits(
    full_db: Path,
    root: Path,
    threads: int,
) -> Path:
    """Build exact current train-vs-train hits as a fallback."""
    out = root / "hits_train_vs_train.parquet"
    if out.exists():
        return out
    query_fasta = root / "db/train_unique.fasta"
    tsv = root / "hits_train_vs_train.tsv"
    _run_blast(query_fasta, full_db, tsv, threads)
    hits = _collapse_top500(_read_hits(tsv), 1e-3)
    hits.to_parquet(out, index=False)
    tsv.unlink()
    print(f"[blast] full train-vs-train complete | hits={len(hits):,}")
    print(f"[blast] wrote: {out}")
    return out


def _incremental_train_hits(
    previous_root: Path,
    current_train: pd.DataFrame,
    previous_train: pd.DataFrame,
    full_db: Path,
    root: Path,
    train_index: pd.DataFrame,
    train_terms_path: Path,
    config: BlastKNNConfig,
    threads: int,
) -> Path:
    """Update train hits from the previous snapshot plus the train delta."""
    out = root / "hits_train_vs_train.parquet"
    if out.exists():
        return out

    previous_path = _previous_hits(
        previous_root, "hits_train_vs_train.parquet"
    )
    if previous_path is None:
        print("[blast] no reusable train hits | running full train-vs-train search")
        return _full_train_hits(full_db, root, threads)

    current_keys = set(current_train["seq_key"].astype(str))
    previous_keys = set(previous_train["seq_key"].astype(str))
    shared_queries = current_keys & previous_keys
    added_subjects = current_keys - previous_keys
    removed_subjects = previous_keys - current_keys
    new_queries = current_keys - previous_keys

    print(f"[blast] reusing train hits: {previous_path}")
    previous_hits = _collapse_top500(
        _read_hits(previous_path), config.evalue_max
    )
    previous_counts = previous_hits.groupby(
        "qseqid", sort=False
    )["sseqid"].nunique()
    saturated = set(
        previous_counts.loc[previous_counts >= 500].index.astype(str)
    )
    removed_hit_queries = set(
        previous_hits.loc[
            previous_hits["sseqid"].isin(removed_subjects), "qseqid"
        ].astype(str)
    )

    previous_dbsize = int(previous_train["length"].sum())
    current_dbsize = int(current_train["length"].sum())
    if previous_dbsize <= 0 or current_dbsize <= 0:
        raise RuntimeError("Invalid BLAST database residue count")

    previous_hits = previous_hits.loc[
        previous_hits["qseqid"].isin(shared_queries)
        & ~previous_hits["sseqid"].isin(removed_subjects)
    ].copy()
    previous_hits["evalue"] = (
        previous_hits["evalue"].to_numpy(dtype=np.float64, copy=False)
        * (current_dbsize / previous_dbsize)
    )

    delta_hits = pd.DataFrame(columns=HIT_COLUMNS)
    if added_subjects and shared_queries:
        added = current_train.loc[
            current_train["seq_key"].isin(added_subjects)
        ].copy()
        shared = current_train.loc[
            current_train["seq_key"].isin(shared_queries)
        ].copy()
        added_fasta = root / "db_added/added_unique.fasta"
        added_db = root / "db_added/added"
        shared_fasta = root / "train_shared.fasta"
        _write_fasta(added, added_fasta)
        print(
            f"[blast] incremental subject DB | added sequences={len(added):,}"
        )
        if _make_db(added_fasta, added_db):
            print(f"[blast] DB ready: {added_db}")
        _write_fasta(shared, shared_fasta)
        delta_tsv = root / "hits_train_added.tsv"
        print(
            f"[blast] search shared queries against added subjects | "
            f"queries={len(shared):,} | subjects={len(added):,}"
        )
        _run_blast(
            shared_fasta, added_db, delta_tsv, threads,
            dbsize=current_dbsize,
        )
        delta_hits = _read_hits(delta_tsv)
        delta_tsv.unlink()
        shared_fasta.unlink()

    merged = pd.concat([previous_hits, delta_hits], ignore_index=True)
    merged = _collapse_top500(merged, config.evalue_max)

    donors = _donor_sets(train_index, train_terms_path)
    seq_keys = np.unique(train_index["seq_key"].to_numpy(copy=False))
    candidates = removed_hit_queries & saturated & shared_queries
    insufficient = _insufficient_oof_queries(
        merged, seq_keys, donors, config, candidates
    )
    fallback_queries = set(new_queries)
    fallback_queries.update(candidates & insufficient)

    if current_dbsize < previous_dbsize:
        fallback_queries.update(shared_queries - saturated)

    if fallback_queries:
        print(f"[blast] full train fallback | queries={len(fallback_queries):,}")
        fallback = current_train.loc[
            current_train["seq_key"].isin(fallback_queries)
        ].copy()
        fallback_fasta = root / "fallback_train.fasta"
        fallback_tsv = root / "hits_train_fallback.tsv"
        _write_fasta(fallback, fallback_fasta)
        _run_blast(fallback_fasta, full_db, fallback_tsv, threads)
        full_hits = _collapse_top500(
            _read_hits(fallback_tsv), config.evalue_max
        )
        merged = _replace_queries(merged, full_hits, fallback_queries)
        merged = _collapse_top500(merged, config.evalue_max)
        fallback_tsv.unlink()
        fallback_fasta.unlink()

    merged.to_parquet(out, index=False)
    fallback_pct = 100.0 * len(fallback_queries) / max(len(current_keys), 1)
    print(
        f"[blast] train-vs-train complete | queries={len(current_keys):,} | "
        f"new={len(new_queries):,} | full fallback={len(fallback_queries):,} "
        f"({fallback_pct:.2f}%) | hits={len(merged):,}"
    )
    print(f"[blast] wrote: {out}")
    return out


def _incremental_test_hits(
    previous_root: Path,
    current_train: pd.DataFrame,
    current_test: pd.DataFrame,
    previous_train: pd.DataFrame,
    previous_test: pd.DataFrame,
    full_db: Path,
    root: Path,
    train_index: pd.DataFrame,
    train_terms_path: Path,
    config: BlastKNNConfig,
    threads: int,
) -> Path:
    """Update test hits from the previous snapshot plus the train delta."""
    out = root / "hits_query.parquet"
    if out.exists():
        return out

    current_train_keys = set(current_train["seq_key"].astype(str))
    previous_train_keys = set(previous_train["seq_key"].astype(str))
    current_test_keys = set(current_test["seq_key"].astype(str))
    previous_test_keys = set(previous_test["seq_key"].astype(str))
    added_subjects = current_train_keys - previous_train_keys
    removed_subjects = previous_train_keys - current_train_keys
    new_queries = current_test_keys - previous_test_keys

    previous_path = _previous_hits(previous_root, "hits_query.parquet")
    if previous_path is None:
        raise FileNotFoundError(
            f"No reusable query hits found in {previous_root}"
        )
    previous_hits = _collapse_top500(
        _read_hits(previous_path), config.evalue_max
    )
    previous_counts = previous_hits.groupby(
        "qseqid", sort=False
    )["sseqid"].nunique()
    saturated = set(
        previous_counts.loc[previous_counts >= 500].index.astype(str)
    )
    removed_hit_queries = set(
        previous_hits.loc[
            previous_hits["sseqid"].isin(removed_subjects), "qseqid"
        ].astype(str)
    )

    previous_dbsize = int(previous_train["length"].sum())
    current_dbsize = int(current_train["length"].sum())
    if previous_dbsize <= 0 or current_dbsize <= 0:
        raise RuntimeError("Invalid BLAST database residue count")

    previous_hits = previous_hits.loc[
        previous_hits["qseqid"].isin(current_test_keys)
        & ~previous_hits["sseqid"].isin(removed_subjects)
    ].copy()
    previous_hits["evalue"] = (
        previous_hits["evalue"].to_numpy(dtype=np.float64, copy=False)
        * (current_dbsize / previous_dbsize)
    )

    delta_hits = pd.DataFrame(columns=HIT_COLUMNS)
    if added_subjects:
        added = current_train.loc[
            current_train["seq_key"].isin(added_subjects)
        ].copy()
        added_fasta = root / "db_added/added_unique.fasta"
        added_db = root / "db_added/added"
        _write_fasta(added, added_fasta)
        _make_db(added_fasta, added_db)
        delta_tsv = root / "hits_query_added.tsv"
        _run_blast(
            root / "db/test_unique.fasta",
            added_db,
            delta_tsv,
            threads,
            dbsize=current_dbsize,
        )
        delta_hits = _read_hits(delta_tsv)
        delta_tsv.unlink()

    merged = pd.concat([previous_hits, delta_hits], ignore_index=True)
    merged = _collapse_top500(merged, config.evalue_max)

    donors = _donor_sets(train_index, train_terms_path)
    insufficient = _insufficient_queries(merged, donors, config)
    fallback_queries = set(new_queries)
    fallback_queries.update(removed_hit_queries & saturated & insufficient)
    if current_dbsize < previous_dbsize:
        shared_queries = current_test_keys & previous_test_keys
        fallback_queries.update(shared_queries - saturated)

    if fallback_queries:
        fallback = current_test.loc[
            current_test["seq_key"].isin(fallback_queries)
        ].copy()
        fallback_fasta = root / "fallback_query.fasta"
        fallback_tsv = root / "hits_query_fallback.tsv"
        _write_fasta(fallback, fallback_fasta)
        _run_blast(fallback_fasta, full_db, fallback_tsv, threads)
        full_hits = _collapse_top500(
            _read_hits(fallback_tsv), config.evalue_max
        )
        merged = _replace_queries(merged, full_hits, fallback_queries)
        merged = _collapse_top500(merged, config.evalue_max)
        fallback_tsv.unlink()
        fallback_fasta.unlink()

    merged.to_parquet(out, index=False)
    fallback_pct = 100.0 * len(fallback_queries) / max(len(current_test_keys), 1)
    print(
        f"[blast] query incremental complete | queries={len(current_test_keys):,} | "
        f"full fallback={len(fallback_queries):,} ({fallback_pct:.2f}%) | "
        f"hits={len(merged):,}"
    )
    print(f"[blast] wrote: {out}")
    return out


def ensure_incremental_blast_train_hits(
    spec,
    config: BlastKNNConfig | None = None,
    threads: int | None = None,
) -> Path:
    config = BlastKNNConfig() if config is None else config
    for executable in ("makeblastdb", "blastp"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required on PATH")

    current_root = spec.prepared_dir.parent
    previous_root = _previous_snapshot(current_root)
    root = current_root / "blast_incremental"
    root.mkdir(parents=True, exist_ok=True)

    train_index = load_index_df(spec.train_index)
    current_train = _unique_sequences(spec.train_index)
    previous_train = _unique_sequences(
        previous_root / "prepared/train_index.parquet"
    )

    current_keys = set(current_train["seq_key"].astype(str))
    previous_keys = set(previous_train["seq_key"].astype(str))
    print(f"[blast] snapshot={previous_root.name} -> {current_root.name}")
    print(
        f"[blast] train sequences={len(current_keys):,} | "
        f"added={len(current_keys - previous_keys):,} | "
        f"removed={len(previous_keys - current_keys):,}"
    )

    train_fasta = root / "db/train_unique.fasta"
    full_db = root / "db/train"
    if not train_fasta.exists():
        _write_fasta(current_train, train_fasta)
        print(f"[blast] wrote: {train_fasta}")
    if _make_db(train_fasta, full_db):
        print(f"[blast] train DB ready: {full_db}")
    else:
        print(f"[blast] using train DB: {full_db}")
    print("[blast] building incremental train-vs-train hits")

    n_threads = int(threads or min(16, os.cpu_count() or 1))
    return _incremental_train_hits(
        previous_root=previous_root,
        current_train=current_train,
        previous_train=previous_train,
        full_db=full_db,
        root=root,
        train_index=train_index,
        train_terms_path=spec.train_terms,
        config=config,
        threads=n_threads,
    )


def ensure_incremental_blast_hits(
    spec,
    config: BlastKNNConfig | None = None,
    threads: int | None = None,
) -> tuple[Path, Path]:
    """Build incrementally updated train and test BLAST hits."""
    config = BlastKNNConfig() if config is None else config
    for executable in ("makeblastdb", "blastp"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required on PATH")

    current_root = spec.prepared_dir.parent
    previous_root = _previous_snapshot(current_root)
    root = current_root / "blast_incremental"
    root.mkdir(parents=True, exist_ok=True)

    train_index = load_index_df(spec.train_index)
    current_train = _unique_sequences(spec.train_index)
    current_test = _unique_sequences(spec.test_index)
    previous_train = _unique_sequences(
        previous_root / "prepared/train_index.parquet"
    )
    previous_test = _unique_sequences(
        previous_root / "prepared/test_index.parquet"
    )

    print(f"[blast] snapshot={previous_root.name} -> {current_root.name}")

    train_fasta = root / "db/train_unique.fasta"
    test_fasta = root / "db/test_unique.fasta"
    full_db = root / "db/train"
    if not train_fasta.exists():
        _write_fasta(current_train, train_fasta)
    if not test_fasta.exists():
        _write_fasta(current_test, test_fasta)
    _make_db(train_fasta, full_db)

    n_threads = int(threads or min(16, os.cpu_count() or 1))
    train_hits = _incremental_train_hits(
        previous_root=previous_root,
        current_train=current_train,
        previous_train=previous_train,
        full_db=full_db,
        root=root,
        train_index=train_index,
        train_terms_path=spec.train_terms,
        config=config,
        threads=n_threads,
    )
    test_hits = _incremental_test_hits(
        previous_root=previous_root,
        current_train=current_train,
        current_test=current_test,
        previous_train=previous_train,
        previous_test=previous_test,
        full_db=full_db,
        root=root,
        train_index=train_index,
        train_terms_path=spec.train_terms,
        config=config,
        threads=n_threads,
    )
    return train_hits, test_hits


def main() -> None:
    spec = dataset()
    config = BlastKNNConfig()
    train_hits = ensure_incremental_blast_train_hits(spec, config=config)
    out_dir = build_blast_oof_component(
        spec, train_hits_path=train_hits, config=config
    )
    print(f"[blast] complete | OOF component={out_dir}")


if __name__ == "__main__":
    main()
