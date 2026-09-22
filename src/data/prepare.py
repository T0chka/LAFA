"""
Prepare the canonical dataset representation consumed by embeddings and models.

Inputs from DatasetSpec:
- current LAFA train FASTA;
- current LAFA query FASTA;
- current LAFA experimental train terms;
- current GO ontology OBO;
- current UniProt-GOA GAF for snapshot-dependent auxiliary features.

Training labels are never supplemented from UniProt-GOA. Information accretion
is computed from the current train terms when no precomputed IA file is supplied.

Outputs in DatasetSpec.prepared_dir:
1) train_index.parquet
   Columns: EntryID, seq_key, length, sequence.
   EntryIDs are defined by the current train_terms.tsv and resolved only from
   the current training FASTA. seq_key is SHA1(sequence).

2) test_index.parquet
   Columns: EntryID, seq_key, length, sequence for every test FASTA record.

3) ground_truth.pkl
   Dict keyed by BPO/CCO/MFO. Each aspect contains a canonical GO term axis,
   parent/child CSR graph, IA weights, propagated training ground truth,
   current train terms, NOT constraints, UniProt experimental and
   non-experimental test terms, and non-experimental evidence-code bit masks.

All GO annotations are canonicalized against the dataset OBO snapshot.
Positive annotations are propagated upward through is_a and part_of parents.
NOT annotations and evidence-code flags are not propagated.

Existing train/test index and ground_truth outputs are reused. Delete an output
to rebuild it.
"""

import hashlib
import pickle
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.csr import pack_row_aligned_csr, pack_row_aligned_csr_with_data
from src.core.debug import print_separator
from src.core.io import load_terms_df
from src.core.ontology import (
    OntologyGraph,
    build_ontology_graph,
    canonize_term,
    closure_csr_by_parents,
    parse_obo_snapshot,
    prune_orphans,
    read_ia_tsv,
)
from src.data.dataset import DatasetSpec
from src.data.gaf import normalize_gaf
from src.data.ia import compute_information_accretion
from src.data.uniprot import ASPECT_TO_NS, update_uniprot_annotations


def _parse_entry_id_from_header(header: str) -> str:
    parts = header.strip().split("|")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    first = header.strip().split()[0]
    if first:
        return first
    raise ValueError(f"Unexpected FASTA header format: {header!r}")


def _iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header = None
    seq = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield _parse_entry_id_from_header(header), "".join(seq)
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if header is not None:
            yield _parse_entry_id_from_header(header), "".join(seq)


def _seq_key(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def _effective_train_terms(spec: DatasetSpec) -> Path:
    return spec.train_terms


def build_indices(spec: DatasetSpec, log_prefix: str = "prepare_data") -> None:
    terms_path = _effective_train_terms(spec)
    terms_df = load_terms_df(terms_path)
    need_ids = set(terms_df["EntryID"].unique().tolist())

    train_rows = []
    test_rows = []
    seq_dict = {}
    found_ids = set()

    for entry_id, seq in _iter_fasta(spec.train_fasta):
        entry_id = str(entry_id)
        if entry_id not in need_ids or entry_id in found_ids:
            continue
        key = _seq_key(seq)
        train_rows.append((entry_id, key, len(seq), seq))
        old = seq_dict.get(key)
        if old is None:
            seq_dict[key] = seq
        elif old != seq:
            raise ValueError(f"Inconsistent sequence for seq_key={key}")
        found_ids.add(entry_id)

    missing = sorted(need_ids - found_ids)
    if missing:
        raise ValueError(
            f"{len(missing)} training EntryIDs have no sequence in train FASTA. "
            f"Sample: {', '.join(missing[:20])}"
        )

    for entry_id, seq in _iter_fasta(spec.test_fasta):
        entry_id = str(entry_id)
        key = _seq_key(seq)
        test_rows.append((entry_id, key, len(seq), seq))
        old = seq_dict.get(key)
        if old is None:
            seq_dict[key] = seq
        elif old != seq:
            raise ValueError(f"Inconsistent sequence for seq_key={key}")

    train_df = pd.DataFrame(
        train_rows, columns=["EntryID", "seq_key", "length", "sequence"]
    )
    test_df = pd.DataFrame(
        test_rows, columns=["EntryID", "seq_key", "length", "sequence"]
    )

    for name, df in (("train", train_df), ("test", test_df)):
        dup = df.groupby("EntryID", sort=False)["seq_key"].nunique()
        bad = dup[dup > 1]
        if not bad.empty:
            raise ValueError(f"{len(bad)} EntryIDs in {name} map to multiple sequences.")

    train_df = train_df.drop_duplicates("EntryID", keep="first")
    test_df = test_df.drop_duplicates("EntryID", keep="first")
    spec.prepared_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(spec.train_index, index=False)
    test_df.to_parquet(spec.test_index, index=False)

    print(
        f"[{log_prefix}] train index | proteins={len(train_df):,} | "
        f"unique sequences={train_df['seq_key'].nunique():,}"
    )
    print(f"[{log_prefix}] wrote train index: {spec.train_index}")
    print(
        f"[{log_prefix}] test index | proteins={len(test_df):,} | "
        f"unique sequences={test_df['seq_key'].nunique():,}"
    )
    print(f"[{log_prefix}] wrote test index: {spec.test_index}")


def _build_entry_csr(
    entry_ids: np.ndarray,
    term_ids: np.ndarray,
    graph: OntologyGraph,
    alt_to_canon: dict[str, str],
    propagate_up: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    term_to_idx = {t: int(i) for i, t in enumerate(graph.term_ids.tolist())}
    row_ids = np.unique(np.asarray(entry_ids, dtype=object))
    term_ids = np.asarray(term_ids, dtype=object)
    col_index = np.fromiter(
        (term_to_idx.get(canonize_term(str(t), alt_to_canon), -1) for t in term_ids),
        dtype=np.int32,
        count=int(term_ids.size),
    )

    n_missing = int((col_index < 0).sum())
    if n_missing:
        raise ValueError(f"{n_missing} terms are missing in the OBO ontology.")

    indptr, indices = pack_row_aligned_csr(
        row_ids_sorted=row_ids,
        entry_ids=np.asarray(entry_ids, dtype=object),
        col_index=col_index,
    )
    if propagate_up:
        indptr, indices = closure_csr_by_parents(
            seed_indptr=indptr,
            seed_indices=indices,
            parents_indptr=graph.parents_indptr,
            parents_indices=graph.parents_indices,
        )
    return row_ids, indptr, indices


def _build_entry_csr_with_data(
    entry_ids: np.ndarray,
    term_ids: np.ndarray,
    data: np.ndarray,
    graph: OntologyGraph,
    alt_to_canon: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    term_to_idx = {t: int(i) for i, t in enumerate(graph.term_ids.tolist())}
    entry_ids = np.asarray(entry_ids, dtype=object)
    term_ids = np.asarray(term_ids, dtype=object)
    data = np.asarray(data)
    row_ids = np.unique(entry_ids)

    col_index = np.fromiter(
        (term_to_idx.get(canonize_term(str(t), alt_to_canon), -1) for t in term_ids),
        dtype=np.int32,
        count=int(term_ids.size),
    )
    n_missing = int((col_index < 0).sum())
    if n_missing:
        raise ValueError(f"{n_missing} terms are missing in the OBO ontology.")

    indptr, indices, out_data = pack_row_aligned_csr_with_data(
        row_ids_sorted=row_ids,
        entry_ids=entry_ids,
        col_index=col_index,
        data=data,
    )
    return row_ids, indptr, indices, out_data


def _empty_csr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, dtype=object),
        np.zeros(1, dtype=np.int64),
        np.empty(0, dtype=np.int32),
    )


def build_ground_truth(spec: DatasetSpec, log_prefix: str = "prepare_data") -> None:
    gt_terms_df = load_terms_df(_effective_train_terms(spec))
    known_terms_df = load_terms_df(spec.train_terms)

    obo = parse_obo_snapshot(str(spec.ontology_obo), list(ASPECT_TO_NS.values()))
    alt_to_canon = obo.alt_to_canon

    ia_map = read_ia_tsv(str(spec.ia_file))
    ia_map_canon = {}
    for term_id, weight in ia_map.items():
        canon = canonize_term(term_id, alt_to_canon)
        if canon in ia_map_canon:
            raise ValueError(f"IA canonicalization collision: {canon}")
        ia_map_canon[canon] = float(weight)

    not_df = (
        pd.read_parquet(spec.not_terms_uniprot)
        if spec.use_uniprot and spec.not_terms_uniprot.exists()
        else None
    )
    test_terms_df = (
        pd.read_parquet(spec.test_terms_uniprot)
        if spec.use_uniprot and spec.test_terms_uniprot.exists()
        else None
    )
    nonexp_df = (
        pd.read_parquet(spec.nonexp_codes_uniprot)
        if spec.use_uniprot and spec.nonexp_codes_uniprot.exists()
        else None
    )

    nonexp_cols = []
    nonexp_code_names = np.empty(0, dtype=object)
    if nonexp_df is not None:
        nonexp_cols = sorted(c for c in nonexp_df.columns if c.startswith("nonexp_"))
        if not nonexp_cols:
            raise ValueError("Non-experimental evidence parquet has no nonexp_* columns.")
        if len(nonexp_cols) > 16:
            raise ValueError("More than 16 non-experimental codes cannot fit uint16.")
        nonexp_code_names = np.asarray(
            [col.removeprefix("nonexp_") for col in nonexp_cols],
            dtype=object,
        )

    prepared = {}
    for aspect, namespace in ASPECT_TO_NS.items():
        graph = build_ontology_graph(
            edges=obo.edges_by_ns[namespace],
            alt_to_canon=alt_to_canon,
        )
        original_terms = int(graph.term_ids.size)
        graph = prune_orphans(graph)
        retained_terms = int(graph.term_ids.size)
        print(
            f"[{log_prefix}] ground truth ontology | {aspect} | original terms={original_terms:,} | "
            f"orphan terms removed={original_terms - retained_terms:,}"
        )
        term_ids = graph.term_ids.astype(object, copy=False)

        ia = np.fromiter(
            (float(ia_map_canon.get(str(term), 0.0)) for term in term_ids),
            dtype=np.float32,
            count=int(term_ids.size),
        )

        block = gt_terms_df.loc[
            gt_terms_df["aspect"] == aspect, ["EntryID", "term"]
        ]
        gt_ids, gt_indptr, gt_indices = _build_entry_csr(
            block["EntryID"].to_numpy(copy=False),
            block["term"].to_numpy(copy=False),
            graph, alt_to_canon, True,
        )

        block = known_terms_df.loc[
            known_terms_df["aspect"] == aspect, ["EntryID", "term"]
        ]
        known_ids, known_indptr, known_indices = _build_entry_csr(
            block["EntryID"].to_numpy(copy=False),
            block["term"].to_numpy(copy=False),
            graph, alt_to_canon, True,
        )

        if not_df is not None:
            block = not_df.loc[not_df["aspect"] == aspect, ["EntryID", "term"]]
            not_ids, not_indptr, not_indices = _build_entry_csr(
                block["EntryID"].to_numpy(copy=False),
                block["term"].to_numpy(copy=False),
                graph, alt_to_canon, False,
            )
        else:
            not_ids, not_indptr, not_indices = _empty_csr()

        if test_terms_df is not None:
            block = test_terms_df.loc[
                test_terms_df["aspect"] == aspect,
                ["EntryID", "term", "is_exp_confirmed"],
            ]
            exp_mask = block["is_exp_confirmed"].to_numpy(dtype=bool, copy=False)

            exp = block.loc[exp_mask, ["EntryID", "term"]]
            test_exp_ids, test_exp_indptr, test_exp_indices = _build_entry_csr(
                exp["EntryID"].to_numpy(copy=False),
                exp["term"].to_numpy(copy=False),
                graph, alt_to_canon, True,
            )

            nonexp = block.loc[~exp_mask, ["EntryID", "term"]]
            test_nonexp_ids, test_nonexp_indptr, test_nonexp_indices = _build_entry_csr(
                nonexp["EntryID"].to_numpy(copy=False),
                nonexp["term"].to_numpy(copy=False),
                graph, alt_to_canon, True,
            )
        else:
            test_exp_ids, test_exp_indptr, test_exp_indices = _empty_csr()
            test_nonexp_ids, test_nonexp_indptr, test_nonexp_indices = _empty_csr()

        if nonexp_df is not None:
            block = nonexp_df.loc[
                nonexp_df["aspect"] == aspect,
                ["EntryID", "term"] + nonexp_cols,
            ]
            grouped = block.groupby(
                ["EntryID", "term"], sort=False, observed=True
            )[nonexp_cols].max()

            mask = np.zeros(len(grouped), dtype=np.uint16)
            for bit, col in enumerate(nonexp_cols):
                values = grouped[col].to_numpy(dtype=np.uint16, copy=False)
                mask |= values << bit

            entry_ids = grouped.index.get_level_values(0).to_numpy(dtype=object, copy=False)
            terms = grouped.index.get_level_values(1).to_numpy(dtype=object, copy=False)
            nonexp_ids, nonexp_indptr, nonexp_indices, nonexp_data = (
                _build_entry_csr_with_data(
                    entry_ids, terms, mask, graph, alt_to_canon
                )
            )
        else:
            nonexp_ids = np.empty(0, dtype=object)
            nonexp_indptr = np.zeros(1, dtype=np.int64)
            nonexp_indices = np.empty(0, dtype=np.int32)
            nonexp_data = np.empty(0, dtype=np.uint16)

        prepared[aspect] = {
            "ontology_term_ids": term_ids,
            "graph": {
                "parents_indptr": graph.parents_indptr.astype(np.int64, copy=False),
                "parents_indices": graph.parents_indices.astype(np.int32, copy=False),
                "children_indptr": graph.children_indptr.astype(np.int64, copy=False),
                "children_indices": graph.children_indices.astype(np.int32, copy=False),
            },
            "ia": ia,
            "gt": {
                "gt_ids": gt_ids,
                "gt_indptr": gt_indptr.astype(np.int64, copy=False),
                "gt_indices": gt_indices.astype(np.int32, copy=False),
            },
            "known_terms": {
                "known_ids": known_ids,
                "known_indptr": known_indptr.astype(np.int64, copy=False),
                "known_indices": known_indices.astype(np.int32, copy=False),
            },
            "not_terms": {
                "not_ids": not_ids,
                "not_indptr": not_indptr.astype(np.int64, copy=False),
                "not_indices": not_indices.astype(np.int32, copy=False),
            },
            "test_exp": {
                "test_exp_ids": test_exp_ids,
                "test_exp_indptr": test_exp_indptr.astype(np.int64, copy=False),
                "test_exp_indices": test_exp_indices.astype(np.int32, copy=False),
            },
            "test_nonexp": {
                "test_nonexp_ids": test_nonexp_ids,
                "test_nonexp_indptr": test_nonexp_indptr.astype(np.int64, copy=False),
                "test_nonexp_indices": test_nonexp_indices.astype(np.int32, copy=False),
            },
            "nonexp_codes": {
                "nonexp_ids": nonexp_ids,
                "nonexp_indptr": nonexp_indptr.astype(np.int64, copy=False),
                "nonexp_indices": nonexp_indices.astype(np.int32, copy=False),
                "nonexp_data": nonexp_data.astype(np.uint16, copy=False),
                "nonexp_code_names": nonexp_code_names,
            },
        }

    with spec.ground_truth.open("wb") as handle:
        pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[{log_prefix}] wrote prepared ground truth: {spec.ground_truth}")
    for aspect, data in prepared.items():
        print(
            f"[{log_prefix}] {aspect} | ontology terms={len(data['ontology_term_ids']):,} | "
            f"train proteins with GO labels={len(data['gt']['gt_ids']):,} | "
            f"NOT proteins={len(data['not_terms']['not_ids']):,} | "
            f"test proteins with experimental GOA annotations={len(data['test_exp']['test_exp_ids']):,} | "
            f"test proteins with non-experimental GOA annotations={len(data['test_nonexp']['test_nonexp_ids']):,} | "
            f"non-experimental feature proteins={len(data['nonexp_codes']['nonexp_ids']):,}"
        )


_MONTH_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _snapshot_key(name: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"([A-Z][a-z]{2})_(\d{4})", name)
    if match is None or match.group(1) not in _MONTH_TO_NUM:
        return None
    return int(match.group(2)), _MONTH_TO_NUM[match.group(1)]


def _find_previous_snapshot(spec: DatasetSpec) -> Path | None:
    current_dir = spec.prepared_dir.parent
    current_key = _snapshot_key(current_dir.name)
    if current_key is None or not current_dir.parent.exists():
        return None

    candidates = []
    for path in current_dir.parent.iterdir():
        key = _snapshot_key(path.name)
        if not path.is_dir() or key is None or key >= current_key:
            continue
        prepared = path / "prepared"
        required = [
            prepared / spec.train_index.name,
            prepared / spec.test_index.name,
            prepared / spec.ground_truth.name,
        ]
        if all(p.exists() for p in required):
            candidates.append((key, path))

    return max(candidates, default=(None, None), key=lambda x: x[0])[1]


def _load_prepared_summary(
    train_index_path: Path,
    test_index_path: Path,
    ground_truth_path: Path,
    train_terms_path: Path,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    train_index = pd.read_parquet(train_index_path, columns=["EntryID", "seq_key"])
    test_index = pd.read_parquet(test_index_path, columns=["EntryID", "seq_key"])
    terms = load_terms_df(train_terms_path)[["EntryID", "term", "aspect"]].drop_duplicates()

    with ground_truth_path.open("rb") as handle:
        ground_truth = pickle.load(handle)

    nonexp_ids = set()
    propagated_annotations = 0
    nonexp_annotations = 0
    for data in ground_truth.values():
        propagated_annotations += int(len(data["gt"]["gt_indices"]))
        nonexp_annotations += int(len(data["nonexp_codes"]["nonexp_indices"]))
        nonexp_ids.update(map(str, data["nonexp_codes"]["nonexp_ids"].tolist()))

    summary = {
        "train proteins": int(train_index["EntryID"].nunique()),
        "test proteins": int(test_index["EntryID"].nunique()),
        "protein-GO rows": int(len(terms)),
        "propagated protein-GO rows": propagated_annotations,
        "training GO terms": int(terms["term"].nunique()),
        "BPO training GO terms": int(terms.loc[terms["aspect"] == "BPO", "term"].nunique()),
        "CCO training GO terms": int(terms.loc[terms["aspect"] == "CCO", "term"].nunique()),
        "MFO training GO terms": int(terms.loc[terms["aspect"] == "MFO", "term"].nunique()),
        "nonexp proteins": int(len(nonexp_ids)),
        "nonexp protein-GO rows": nonexp_annotations,
    }
    return summary, train_index, test_index


def write_snapshot_comparison(spec: DatasetSpec, log_prefix: str = "prepare_data") -> None:
    current_dir = spec.prepared_dir.parent
    previous_dir = _find_previous_snapshot(spec)
    if previous_dir is None:
        print(f"[{log_prefix}] snapshot comparison skipped: no earlier prepared snapshot found")
        return

    previous_terms = spec.train_terms.parent.parent / previous_dir.name / spec.train_terms.name
    if not previous_terms.exists():
        print(
            f"[{log_prefix}] snapshot comparison skipped: previous train terms not found: "
            f"{previous_terms}"
        )
        return

    previous_prepared = previous_dir / "prepared"
    previous_summary, previous_train, previous_test = _load_prepared_summary(
        previous_prepared / spec.train_index.name,
        previous_prepared / spec.test_index.name,
        previous_prepared / spec.ground_truth.name,
        previous_terms,
    )
    current_summary, current_train, current_test = _load_prepared_summary(
        spec.train_index, spec.test_index, spec.ground_truth, spec.train_terms
    )

    rows = []
    for metric, previous in previous_summary.items():
        current = current_summary[metric]
        delta = current - previous
        delta_pct = 100.0 * delta / previous if previous else np.nan
        rows.append((metric, previous, current, delta, delta_pct))

    comparison = pd.DataFrame(
        rows, columns=["metric", "previous", "current", "delta", "delta_pct"]
    )

    def seq_map(df: pd.DataFrame) -> dict[str, str]:
        return dict(zip(df["EntryID"].astype(str), df["seq_key"].astype(str)))

    previous_train_map = seq_map(previous_train)
    current_train_map = seq_map(current_train)
    previous_test_map = seq_map(previous_test)
    current_test_map = seq_map(current_test)

    previous_train_ids = set(previous_train_map)
    current_train_ids = set(current_train_map)
    previous_test_ids = set(previous_test_map)
    current_test_ids = set(current_test_map)

    previous_terms_df = load_terms_df(previous_terms)[["EntryID", "term"]].drop_duplicates()
    current_terms_df = load_terms_df(spec.train_terms)[["EntryID", "term"]].drop_duplicates()
    previous_pairs = set(map(tuple, previous_terms_df.astype(str).to_numpy()))
    current_pairs = set(map(tuple, current_terms_df.astype(str).to_numpy()))
    previous_go = set(previous_terms_df["term"].astype(str))
    current_go = set(current_terms_df["term"].astype(str))

    changed_train = sum(
        previous_train_map[entry_id] != current_train_map[entry_id]
        for entry_id in previous_train_ids & current_train_ids
    )
    changed_test = sum(
        previous_test_map[entry_id] != current_test_map[entry_id]
        for entry_id in previous_test_ids & current_test_ids
    )

    changes = pd.DataFrame(
        [
            ("train proteins added", len(current_train_ids - previous_train_ids)),
            ("train proteins removed", len(previous_train_ids - current_train_ids)),
            ("test proteins added", len(current_test_ids - previous_test_ids)),
            ("test proteins removed", len(previous_test_ids - current_test_ids)),
            ("shared train proteins with changed sequence", changed_train),
            ("shared test proteins with changed sequence", changed_test),
            ("unique protein-GO rows added", len(current_pairs - previous_pairs)),
            ("unique protein-GO rows removed", len(previous_pairs - current_pairs)),
            ("training GO terms added", len(current_go - previous_go)),
            ("training GO terms removed", len(previous_go - current_go)),
        ],
        columns=["change", "count"],
    )

    comparison_path = current_dir / "snapshot_comparison.tsv"
    changes_path = current_dir / "snapshot_changes.tsv"
    comparison.to_csv(comparison_path, sep="\t", index=False, float_format="%.3f")
    changes.to_csv(changes_path, sep="\t", index=False)

    display = comparison.copy()
    display["delta_pct"] = display["delta_pct"].map(
        lambda x: "" if pd.isna(x) else f"{x:+.2f}%"
    )
    for col in ["previous", "current"]:
        display[col] = display[col].map(lambda x: f"{int(x):,}")
    display["delta"] = display["delta"].map(lambda x: f"{int(x):+,}")

    print_separator(log_prefix, f"Snapshot comparison: {previous_dir.name} -> {current_dir.name}")
    print(display.to_string(index=False))
    print(f"[{log_prefix}] snapshot composition changes")
    changes_display = changes.copy()
    changes_display["count"] = changes_display["count"].map(lambda x: f"{int(x):+,}")
    print(changes_display.to_string(index=False))
    print(f"[{log_prefix}] wrote: {comparison_path}")
    print(f"[{log_prefix}] wrote: {changes_path}")


def prepare_dataset(spec: DatasetSpec, log_prefix: str = "prepare_data") -> None:
    snapshot = spec.prepared_dir.parent.name
    print(f"[{log_prefix}] snapshot={snapshot}")
    spec.prepared_dir.mkdir(parents=True, exist_ok=True)
    spec.cache_dir.mkdir(parents=True, exist_ok=True)

    print_separator(log_prefix, "Information accretion")
    if not spec.ia_file.exists():
        print(f"[{log_prefix}] IA not found; computing from train protein-GO rows and GO ontology")
        compute_information_accretion(
            spec.train_terms, spec.ontology_obo, spec.ia_file, log_prefix=log_prefix
        )
    else:
        print(f"[{log_prefix}] using existing IA: {spec.ia_file}")

    if spec.use_uniprot:
        print_separator(log_prefix, "UniProt-GOA auxiliary data")
        if not spec.uniprot_gaf_parquet.exists():
            if spec.uniprot_gaf_gz is None or not spec.uniprot_gaf_gz.exists():
                raise FileNotFoundError(
                    "UniProt snapshot is missing. Provide either the normalized "
                    "Parquet or the raw .gaf.gz file declared by DatasetSpec."
                )
            print(f"[{log_prefix}] normalized GAF not found; converting raw UniProt-GOA GAF")
            normalize_gaf(
                spec.uniprot_gaf_gz, spec.uniprot_gaf_parquet, log_prefix=log_prefix
            )
        else:
            print(f"[{log_prefix}] using existing normalized GAF: {spec.uniprot_gaf_parquet}")
        print(f"[{log_prefix}] building snapshot-specific UniProt features for train and test proteins")
        update_uniprot_annotations(spec, log_prefix=log_prefix)

    print_separator(log_prefix, "Train and test sequence indices")
    if not spec.train_index.exists() or not spec.test_index.exists():
        build_indices(spec, log_prefix=log_prefix)
    else:
        print(f"[{log_prefix}] using existing train index: {spec.train_index}")
        print(f"[{log_prefix}] using existing test index: {spec.test_index}")

    print_separator(log_prefix, "Ontology and training labels")
    if not spec.ground_truth.exists():
        build_ground_truth(spec, log_prefix=log_prefix)
    else:
        print(f"[{log_prefix}] using existing prepared ground truth: {spec.ground_truth}")

    print_separator(log_prefix, "Prepared snapshot summary")
    summary, _, _ = _load_prepared_summary(
        spec.train_index, spec.test_index, spec.ground_truth, spec.train_terms
    )
    print(
        f"[{log_prefix}] train proteins={summary['train proteins']:,} | "
        f"test proteins={summary['test proteins']:,}"
    )
    print(
        f"[{log_prefix}] protein-GO rows={summary['protein-GO rows']:,} | "
        f"propagated protein-GO rows={summary['propagated protein-GO rows']:,}"
    )
    print(
        f"[{log_prefix}] training GO terms={summary['training GO terms']:,} | "
        f"BPO={summary['BPO training GO terms']:,} | "
        f"CCO={summary['CCO training GO terms']:,} | "
        f"MFO={summary['MFO training GO terms']:,}"
    )
    print(
        f"[{log_prefix}] non-experimental proteins={summary['nonexp proteins']:,} | "
        f"non-experimental protein-GO rows={summary['nonexp protein-GO rows']:,}"
    )
    write_snapshot_comparison(spec, log_prefix=log_prefix)
