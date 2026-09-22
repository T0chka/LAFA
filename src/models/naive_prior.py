from pathlib import Path

import numpy as np
import pandas as pd

from src.core.io import load_index_df, load_prepared_gt, load_terms_df
from src.core.postprocess import CSRState, PostprocessConfig, Postprocessor
from src.data.dataset import DatasetSpec


POSTPROCESS = PostprocessConfig(top_k=500, min_score=0.0, drop_zero_ia=True, exclude_not_descendants=True, drop_weak_preds=False)


def _terms_path(dataset: DatasetSpec) -> Path:
    return dataset.train_terms


def build_term_priors(train_terms: pd.DataFrame, topk_by_aspect: dict[str, int]) -> pd.DataFrame:
    train_terms = train_terms.loc[:, ["EntryID", "term", "aspect"]].drop_duplicates()
    n_proteins = train_terms.loc[:, ["aspect", "EntryID"]].drop_duplicates().groupby("aspect", sort=False).size().astype(np.float64, copy=False)
    counts = train_terms.groupby(["aspect", "term"], sort=False).size().rename("cnt").reset_index()
    denom = counts["aspect"].map(n_proteins).to_numpy(dtype=np.float64, copy=False)
    counts["score"] = (counts["cnt"].to_numpy(copy=False) / denom).astype(np.float32, copy=False)
    out = []
    for aspect, block in counts.groupby("aspect", sort=False):
        out.append(block.sort_values("score", ascending=False, kind="mergesort").head(int(topk_by_aspect[str(aspect)])))
    return pd.concat(out, ignore_index=True).loc[:, ["aspect", "term", "score"]]


def _state(entry_ids: np.ndarray, term_pos: np.ndarray, scores: np.ndarray) -> CSRState:
    order = np.argsort(term_pos, kind="mergesort")
    term_pos = term_pos[order]
    scores = scores[order]
    n_rows = int(entry_ids.size)
    n_terms = int(term_pos.size)
    indptr = np.arange(n_rows + 1, dtype=np.int64) * n_terms
    return CSRState(indptr=indptr, indices=np.tile(term_pos, n_rows).astype(np.int32, copy=False), scores=np.tile(scores, n_rows).astype(np.float32, copy=False))


def _save(
    index_df: pd.DataFrame,
    priors: pd.DataFrame,
    prepared_gt: dict,
    data_type: str,
    out_dir: Path,
    log_prefix: str,
) -> None:
    entry_ids = pd.unique(index_df["EntryID"]).astype(object, copy=False)
    postprocess = Postprocessor(POSTPROCESS, index_df, prepared_gt)
    prefix = "oof" if data_type == "oof" else "submit"
    target = out_dir / prefix
    target.mkdir(parents=True, exist_ok=True)
    for aspect, aspect_gt in prepared_gt.items():
        block = priors.loc[priors["aspect"] == aspect]
        term_pos = postprocess.map_terms_to_pos(aspect, block["term"].to_numpy(dtype=object, copy=False))
        state = _state(entry_ids, term_pos, block["score"].to_numpy(dtype=np.float32, copy=False))
        label = "OOF" if data_type == "oof" else "prediction"
        print(f"[{log_prefix}] naive_prior | {label} | {aspect}")
        topk_pos, topk_scores = postprocess.postprocess_state(
            state=state, data_type=data_type, entry_ids=entry_ids, aspect_name=aspect,
            propagate=False, add_nonexp_terms=False, add_exp_terms=False,
            drop_known=(data_type == "test"),
            log_context=None,
            log_prefix=log_prefix,
        )
        out_path = target / f"{prefix}_for_ltr_{aspect}.npz"
        np.savez_compressed(out_path, entry_ids=entry_ids, term_pos=topk_pos, scores=topk_scores)
        print(f"[{log_prefix}] wrote {label}: {out_path}")
        print()


def _priors(dataset: DatasetSpec) -> tuple[pd.DataFrame, dict]:
    train_index = load_index_df(dataset.train_index)
    train_terms = load_terms_df(_terms_path(dataset))
    prepared_gt = load_prepared_gt(dataset.ground_truth)
    postprocess = Postprocessor(POSTPROCESS, train_index, prepared_gt)
    priors = build_term_priors(train_terms, postprocess._topk_by_aspect)
    return priors, prepared_gt


def build_naive_oof(
    dataset: DatasetSpec,
    out_dir: str | Path | None = None,
    log_prefix: str = "build_naive",
) -> Path:
    out_dir = dataset.prepared_dir.parent / "predictors/naive_prior" if out_dir is None else Path(out_dir)
    priors, prepared_gt = _priors(dataset)
    train_index = load_index_df(dataset.train_index)
    counts = priors.groupby("aspect", sort=False).size().to_dict()
    print(
        f"[{log_prefix}] OOF predictions for LTR | proteins={train_index['EntryID'].nunique():,} | "
        f"prior terms: BPO={counts.get('BPO', 0):,} CCO={counts.get('CCO', 0):,} MFO={counts.get('MFO', 0):,}"
    )
    _save(train_index, priors, prepared_gt, "oof", out_dir, log_prefix)
    return out_dir


def predict_naive_component(
    dataset: DatasetSpec,
    index_df: pd.DataFrame,
    out_dir: str | Path,
    log_prefix: str = "predict",
) -> Path:
    out_dir = Path(out_dir)
    priors, prepared_gt = _priors(dataset)
    print(f"[{log_prefix}] component predictions | proteins={index_df['EntryID'].nunique():,}")
    _save(index_df, priors, prepared_gt, "test", out_dir, log_prefix)
    return out_dir


def build_naive_component(
    dataset: DatasetSpec,
    out_dir: str | Path | None = None,
    log_prefix: str = "build_naive",
) -> Path:
    out_dir = build_naive_oof(dataset, out_dir, log_prefix=log_prefix)
    predict_naive_component(
        dataset, load_index_df(dataset.test_index), out_dir, log_prefix=log_prefix
    )
    return out_dir
