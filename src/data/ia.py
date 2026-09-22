"""Compute information accretion from the current training snapshot."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.csr import invert_csr, pack_row_aligned_csr
from src.core.io import load_terms_df
from src.core.ontology import build_ontology_graph, canonize_term, closure_csr_by_parents, parse_obo_snapshot, prune_orphans


ASPECT_TO_NS = {
    "BPO": "biological_process",
    "CCO": "cellular_component",
    "MFO": "molecular_function",
}


def compute_information_accretion(
    train_terms: Path,
    ontology_obo: Path,
    out_path: Path,
    log_prefix: str = "prepare_data",
) -> Path:
    """Compute term IA from current experimental train labels and GO graph."""
    terms_df = load_terms_df(train_terms)
    obo = parse_obo_snapshot(str(ontology_obo), list(ASPECT_TO_NS.values()))
    rows = []

    for aspect, namespace in ASPECT_TO_NS.items():
        raw_graph = build_ontology_graph(obo.edges_by_ns[namespace], obo.alt_to_canon)
        original_terms = int(raw_graph.term_ids.size)
        graph = prune_orphans(raw_graph)
        retained_terms = int(graph.term_ids.size)
        print(
            f"[{log_prefix}] IA ontology | {aspect} | original terms={original_terms:,} | "
            f"orphan terms removed={original_terms - retained_terms:,}"
        )
        term_to_idx = {str(term): i for i, term in enumerate(graph.term_ids.tolist())}
        block = terms_df.loc[terms_df["aspect"] == aspect, ["EntryID", "term"]].copy()
        block["term"] = block["term"].map(lambda x: canonize_term(str(x), obo.alt_to_canon))
        block = block.loc[block["term"].isin(term_to_idx)].drop_duplicates()

        entry_ids = np.unique(block["EntryID"].to_numpy(dtype=object, copy=False))
        col_index = np.fromiter(
            (term_to_idx[str(term)] for term in block["term"]),
            dtype=np.int32,
            count=len(block),
        )
        indptr, indices = pack_row_aligned_csr(
            entry_ids,
            block["EntryID"].to_numpy(dtype=object, copy=False),
            col_index,
        )
        indptr, indices = closure_csr_by_parents(
            indptr, indices, graph.parents_indptr, graph.parents_indices
        )
        term_indptr, term_proteins = invert_csr(indptr, indices, len(graph.term_ids))
        counts = term_indptr[1:] - term_indptr[:-1]

        for term_idx, term_id in enumerate(graph.term_ids.tolist()):
            ps = int(graph.parents_indptr[term_idx])
            pe = int(graph.parents_indptr[term_idx + 1])
            parents = graph.parents_indices[ps:pe]
            if parents.size == 0:
                ia = 0.0
            elif parents.size == 1:
                parent_count = int(counts[int(parents[0])])
                ia = -np.log2((int(counts[term_idx]) + 1.0) / (parent_count + 1.0))
            else:
                parent_sets = []
                for parent in parents:
                    s = int(term_indptr[int(parent)])
                    e = int(term_indptr[int(parent) + 1])
                    parent_sets.append(term_proteins[s:e])
                common = parent_sets[0]
                for proteins in parent_sets[1:]:
                    common = np.intersect1d(common, proteins, assume_unique=True)
                ia = -np.log2((int(counts[term_idx]) + 1.0) / (int(common.size) + 1.0))
            rows.append((str(term_id), float(max(0.0, ia))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["term", "ia"]).to_csv(out_path, sep="\t", index=False)
    print(f"[{log_prefix}] wrote IA: {out_path}")
    return out_path
