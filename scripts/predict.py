import argparse
from pathlib import Path

import pandas as pd

from config import dataset
from scripts.build_hmlp import SPEC as HMLP_SPEC
from scripts.build_mlp import SPEC as MLP_SPEC
from scripts.build_pyboost import SPEC as PYBOOST_SPEC
from src.core.debug import print_separator
from src.core.io import load_index_df
from src.ltr.ranker import LTRMember, predict_ltr
from src.models.blast_knn import predict_blast_component
from src.models.blast_search import build_blast_query_hits
from src.models.naive_prior import predict_naive_component
from src.models.nonexp import predict_nonexp_component
from src.models.predictor import predict_predictor


MEMBER_NAMES = (
    "hmlp_esm2",
    "mlp_t5_esm1b",
    "pyb_t5",
    "blast_knn",
    "naive_prior",
    "nonexp",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets")
    parser.add_argument("--prediction-work")
    parser.add_argument("--num-threads", type=int, default=None)
    return parser


def _target_ids(path: str | Path) -> list[str]:
    ids = []
    seen = set()
    for line in Path(path).read_text().splitlines():
        value = line.strip().split("\t", 1)[0].strip()
        if not value or value == "EntryID" or value in seen:
            continue
        seen.add(value)
        ids.append(value)
    if not ids:
        raise ValueError(f"No target EntryID values found in {path}")
    return ids


def _prediction_index(ds, targets: str | None) -> pd.DataFrame:
    index_df = load_index_df(ds.test_index)
    if targets is None:
        return index_df
    ids = _target_ids(targets)
    wanted = set(ids)
    available = set(index_df["EntryID"].astype(str))
    missing = [entry_id for entry_id in ids if entry_id not in available]
    if missing:
        sample = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} targets are absent from test_index: {sample}")
    return index_df.loc[index_df["EntryID"].astype(str).isin(wanted)].copy()


def _check_manifest(prediction_work: Path, index_df: pd.DataFrame) -> None:
    ids = sorted(set(index_df["EntryID"].astype(str)))
    manifest = prediction_work / "target_entry_ids.txt"
    current = "\n".join(ids) + "\n"
    if manifest.exists() and manifest.read_text() != current:
        raise RuntimeError(
            f"Prediction work directory contains a different target set: {prediction_work}"
        )
    prediction_work.mkdir(parents=True, exist_ok=True)
    manifest.write_text(current)


def main() -> None:
    args = _parser().parse_args()
    ds = dataset()
    source_work = ds.prepared_dir.parent
    if args.targets and not args.prediction_work:
        raise ValueError("--prediction-work is required when --targets is used")
    prediction_work = source_work if args.prediction_work is None else Path(args.prediction_work)
    index_df = _prediction_index(ds, args.targets)
    _check_manifest(prediction_work, index_df)

    n_ids = index_df["EntryID"].nunique()
    n_seq = index_df["seq_key"].nunique()
    source_snapshot = source_work.name
    print_separator("predict", "Prediction run", char="=")
    print(f"[predict] source={source_snapshot}")
    if args.targets:
        requested = len(_target_ids(args.targets))
        print(f"[predict] targets file: {Path(args.targets)}")
        print(
            f"[predict] requested proteins={requested:,} | matched proteins={n_ids:,} | "
            f"unique_sequences={n_seq:,}"
        )
    else:
        print(f"[predict] mode=full test set | proteins={n_ids:,} | unique_sequences={n_seq:,}")
    print(f"[predict] source work: {source_work}")
    print(f"[predict] output: {prediction_work}")

    for spec in (HMLP_SPEC, MLP_SPEC, PYBOOST_SPEC):
        print_separator("predict", f"{spec.name} predictions")
        source_dir = source_work / "predictors" / spec.name
        save_dir = prediction_work / "predictors" / spec.name
        predict_predictor(
            ds, spec, model_dir=source_dir, index_df=index_df, save_dir=save_dir,
            log_prefix="predict",
        )

    print_separator("predict", "BLAST-KNN predictions")
    query_hits = build_blast_query_hits(
        ds,
        index_df,
        prediction_work / "blast_query",
        threads=args.num_threads,
        log_prefix="predict",
    )
    predict_blast_component(
        ds,
        query_hits,
        index_df,
        prediction_work / "predictors/blast_knn",
        log_prefix="predict",
    )
    print_separator("predict", "Naive-prior predictions")
    predict_naive_component(
        ds,
        index_df,
        prediction_work / "predictors/naive_prior",
        log_prefix="predict",
    )
    print_separator("predict", "Non-experimental predictions")
    predict_nonexp_component(
        ds,
        index_df,
        prediction_work / "predictors/nonexp",
        log_prefix="predict",
    )

    members = [
        LTRMember(name, prediction_work / "predictors" / name)
        for name in MEMBER_NAMES
    ]
    print_separator("predict", "Final LTR ensemble prediction")
    predict_ltr(
        ds,
        members=members,
        model_dir=source_work / "ltr/models",
        out_dir=prediction_work / "final",
        index_df=index_df,
        log_prefix="predict",
    )


if __name__ == "__main__":
    main()
