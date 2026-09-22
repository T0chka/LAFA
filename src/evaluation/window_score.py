from pathlib import Path

import numpy as np
import pandas as pd

from src.core.csr import pack_row_aligned_csr
from src.core.debug import print_separator
from src.core.io import load_prepared_gt
from src.core.scoring import score_weighted_f


METHODS = (
    "hmlp_esm2",
    "mlp_t5_esm1b",
    "pyb_t5",
    "blast_knn",
    "naive_prior",
    "nonexp",
    "ltr",
)
REGIMES = ("NK", "LK", "PK")
ASPECTS = ("BPO", "CCO", "MFO")
DEMOCAFA_ASPECTS = {
    "P": "BPO",
    "C": "CCO",
    "F": "MFO",
    "BPO": "BPO",
    "CCO": "CCO",
    "MFO": "MFO",
}


def _read_terms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"EntryID", "term", "aspect"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[["EntryID", "term", "aspect"]].dropna().drop_duplicates().copy()
    mapped = df["aspect"].map(DEMOCAFA_ASPECTS)
    if mapped.isna().any():
        bad = sorted(df.loc[mapped.isna(), "aspect"].unique())
        raise ValueError(f"{path} has unsupported aspect values: {bad}")
    df["aspect"] = mapped
    return df


def _read_targets(path: Path) -> set[str]:
    targets = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    targets.discard("EntryID")
    return targets


def _load_groundtruth(
    groundtruth_dir: Path,
    source_gt: dict,
    log_prefix: str = "score_eval",
) -> dict[str, dict[str, dict]]:
    paths = {regime: groundtruth_dir / f"groundtruth_{regime}.tsv" for regime in REGIMES}
    required = list(paths.values()) + [
        groundtruth_dir / "groundtruth_PK_known.tsv",
        groundtruth_dir / "groundtruth_terms_of_interest.txt",
        groundtruth_dir / "groundtruth_targets.tsv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing democafa outputs:\n" + "\n".join(map(str, missing)))

    frames = {regime: _read_terms(path) for regime, path in paths.items()}
    toi = {
        line.strip()
        for line in (groundtruth_dir / "groundtruth_terms_of_interest.txt").read_text().splitlines()
        if line.strip()
    }
    if not toi:
        raise ValueError("groundtruth_terms_of_interest.txt is empty")

    targets = _read_targets(groundtruth_dir / "groundtruth_targets.tsv")
    union = set().union(*(set(df["EntryID"]) for df in frames.values()))
    if targets != union:
        raise ValueError(
            "groundtruth_targets.tsv does not match NK/LK/PK union: "
            f"targets={len(targets):,}, union={len(union):,}, "
            f"targets_only={len(targets - union):,}, union_only={len(union - targets):,}"
        )

    out = {regime: {} for regime in REGIMES}
    for aspect in ASPECTS:
        base = source_gt[aspect]
        term_ids = base["ontology_term_ids"].astype(object, copy=False)
        term_to_pos = {str(term): i for i, term in enumerate(term_ids)}
        ia = base["ia"].astype(np.float32, copy=True)
        keep_toi = np.fromiter((str(term) in toi for term in term_ids), dtype=bool, count=len(term_ids))
        ia[~keep_toi] = np.float32(0.0)

        for regime in REGIMES:
            block = frames[regime].loc[frames[regime]["aspect"] == aspect, ["EntryID", "term"]]
            unknown = sorted(set(block["term"]) - set(term_to_pos))
            if unknown:
                sample = ", ".join(unknown[:10])
                raise ValueError(
                    f"{regime} {aspect}: {len(unknown)} GT terms are absent from source ontology: {sample}"
                )

            row_ids = np.asarray(sorted(block["EntryID"].unique()), dtype=object)
            col_index = block["term"].map(term_to_pos).to_numpy(dtype=np.int32, copy=False)
            indptr, indices = pack_row_aligned_csr(
                row_ids,
                block["EntryID"].to_numpy(dtype=object, copy=False),
                col_index,
            )
            out[regime][aspect] = {
                "ontology_term_ids": term_ids,
                "graph": base["graph"],
                "ia": ia,
                "gt": {
                    "gt_ids": row_ids,
                    "gt_indptr": indptr.astype(np.int64, copy=False),
                    "gt_indices": indices.astype(np.int32, copy=False),
                },
            }

    print_separator(log_prefix, "Ground-truth composition")
    for regime in REGIMES:
        counts = []
        for aspect in ASPECTS:
            n = int(out[regime][aspect]["gt"]["gt_ids"].size)
            counts.append(f"{aspect}={n:,}")
        print(
            f"[{log_prefix}] {regime} | proteins={frames[regime]['EntryID'].nunique():,} | "
            + " | ".join(counts)
        )
    print(f"[{log_prefix}] unique targets={len(targets):,} | terms_of_interest={len(toi):,}")
    return out


def _target_ids(gt_by_regime: dict, aspect: str) -> np.ndarray:
    ids = []
    for regime in REGIMES:
        ids.extend(gt_by_regime[regime][aspect]["gt"]["gt_ids"].tolist())
    return np.asarray(sorted(set(map(str, ids))), dtype=object)


def _load_component(source_work: Path, method: str, aspect: str, wanted_ids: np.ndarray):
    path = source_work / "predictors" / method / "submit" / f"submit_for_ltr_{aspect}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    wanted = set(map(str, wanted_ids))
    with np.load(path, allow_pickle=True) as data:
        entry_ids = data["entry_ids"].astype(object, copy=False)
        keep = np.fromiter((str(x) in wanted for x in entry_ids), dtype=bool, count=len(entry_ids))
        ids = entry_ids[keep].copy()
        term_pos = data["term_pos"][keep].astype(np.int32, copy=True)
        scores = data["scores"][keep].astype(np.float32, copy=True)
    return ids, term_pos, scores


def _load_ltr(source_work: Path, source_gt: dict, wanted_by_aspect: dict[str, np.ndarray]):
    path = source_work / "final" / "submission.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    wanted_all = set()
    for ids in wanted_by_aspect.values():
        wanted_all.update(map(str, ids))
    parts = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["EntryID", "term", "score"],
        dtype={"EntryID": str, "term": str, "score": np.float32},
        chunksize=1_000_000,
    ):
        keep = chunk["EntryID"].isin(wanted_all)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["EntryID", "term", "score"])

    out = {}
    for aspect in ASPECTS:
        wanted = set(map(str, wanted_by_aspect[aspect]))
        term_ids = source_gt[aspect]["ontology_term_ids"].astype(str, copy=False)
        term_to_pos = {term: i for i, term in enumerate(term_ids)}
        x = df.loc[df["EntryID"].isin(wanted) & df["term"].isin(term_to_pos), ["EntryID", "term", "score"]].copy()
        x["term_pos"] = x["term"].map(term_to_pos).astype(np.int32)
        entry_ids = pd.unique(x["EntryID"]).astype(object, copy=False)
        if not len(entry_ids):
            out[aspect] = (entry_ids, np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.float32))
            continue
        groups = {entry_id: g for entry_id, g in x.groupby("EntryID", sort=False)}
        max_k = max(len(g) for g in groups.values())
        term_pos = np.full((len(entry_ids), max_k), -1, dtype=np.int32)
        scores = np.zeros((len(entry_ids), max_k), dtype=np.float32)
        for i, entry_id in enumerate(entry_ids):
            g = groups[entry_id]
            order = np.argsort(-g["score"].to_numpy(dtype=np.float32), kind="stable")
            pos = g["term_pos"].to_numpy(dtype=np.int32)[order]
            val = g["score"].to_numpy(dtype=np.float32)[order]
            n = len(g)
            term_pos[i, :n] = pos
            scores[i, :n] = val
        out[aspect] = (entry_ids, term_pos, scores)
    return out


def _best_f(gt: dict, entry_ids: np.ndarray, term_pos: np.ndarray, scores: np.ndarray, aspect: str, th_step: float):
    metrics = score_weighted_f(
        {aspect: gt[aspect]},
        {aspect: (entry_ids, term_pos)},
        {aspect: scores},
        th_step=th_step,
        eval_on="gt",
    )
    if metrics.empty:
        raise RuntimeError(f"No scorable rows for {aspect}")
    best = metrics.loc[metrics["f_micro"].idxmax()]
    return float(best["tau"]), float(best["f_micro"]), int(round(float(best["n"])))


def score_window(
    groundtruth_dir: str | Path,
    source_work: str | Path,
    source_snapshot: str,
    future_snapshot: str,
    output_dir: str | Path | None = None,
    th_step: float = 0.01,
    prediction_work: str | Path | None = None,
    log_prefix: str = "score_eval",
):
    groundtruth_dir = Path(groundtruth_dir)
    source_work = Path(source_work)
    prediction_work = source_work if prediction_work is None else Path(prediction_work)
    output_dir = groundtruth_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{log_prefix}] window={source_snapshot} -> {future_snapshot}")
    print(f"[{log_prefix}] predictions: {prediction_work}")
    print(f"[{log_prefix}] ground truth: {groundtruth_dir}")
    print(f"[{log_prefix}] threshold step={th_step:g}")
    source_gt = load_prepared_gt(source_work / "prepared" / "ground_truth.pkl")
    gt_by_regime = _load_groundtruth(groundtruth_dir, source_gt, log_prefix=log_prefix)
    wanted_by_aspect = {aspect: _target_ids(gt_by_regime, aspect) for aspect in ASPECTS}
    ltr = _load_ltr(prediction_work, source_gt, wanted_by_aspect)

    rows = []
    for method in METHODS:
        for aspect in ASPECTS:
            if method == "ltr":
                entry_ids, term_pos, scores = ltr[aspect]
            else:
                entry_ids, term_pos, scores = _load_component(prediction_work, method, aspect, wanted_by_aspect[aspect])
            for regime in REGIMES:
                gt = gt_by_regime[regime]
                tau, f_micro, covered = _best_f(gt, entry_ids, term_pos, scores, aspect, th_step)
                n_targets = int(gt[aspect]["gt"]["gt_ids"].size)
                rows.append((source_snapshot, future_snapshot, method, regime, aspect, n_targets, covered, tau, f_micro))

    long = pd.DataFrame(
        rows,
        columns=[
            "source_snapshot",
            "future_snapshot",
            "method",
            "regime",
            "aspect",
            "targets",
            "covered_at_tau",
            "tau",
            "f_micro",
        ],
    )
    long.to_csv(output_dir / "scores_long.tsv", sep="\t", index=False, float_format="%.6f")

    wide = long.pivot(index="method", columns=["regime", "aspect"], values="f_micro")
    wide = wide.reindex(index=METHODS, columns=pd.MultiIndex.from_product([REGIMES, ASPECTS]))
    wide.columns = [f"{regime}_{aspect}" for regime, aspect in wide.columns]
    wide["mean"] = wide.mean(axis=1)
    wide.insert(0, "future_snapshot", future_snapshot)
    wide.insert(0, "source_snapshot", source_snapshot)
    wide = wide.reset_index()
    wide.to_csv(output_dir / "scores.tsv", sep="\t", index=False, float_format="%.6f")

    tau = long.pivot(index="method", columns=["regime", "aspect"], values="tau")
    tau = tau.reindex(index=METHODS, columns=pd.MultiIndex.from_product([REGIMES, ASPECTS]))
    tau.columns = [f"{regime}_{aspect}" for regime, aspect in tau.columns]
    tau.insert(0, "future_snapshot", future_snapshot)
    tau.insert(0, "source_snapshot", source_snapshot)
    tau.reset_index().to_csv(output_dir / "thresholds.tsv", sep="\t", index=False, float_format="%.2f")

    print_separator(log_prefix, "F-micro results")
    print(wide.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"[{log_prefix}] wrote: {output_dir / 'scores_long.tsv'}")
    print(f"[{log_prefix}] wrote: {output_dir / 'scores.tsv'}")
    print(f"[{log_prefix}] wrote: {output_dir / 'thresholds.tsv'}")
    return long, wide
