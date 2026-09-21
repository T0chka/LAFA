"""Prepare snapshot-specific UniProt-GOA features without modifying train labels."""

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd

from src.core.io import load_terms_df
from src.core.ontology import parse_obo_snapshot
from src.data.dataset import DatasetSpec


ASPECT_TO_NS = {
    "BPO": "biological_process",
    "CCO": "cellular_component",
    "MFO": "molecular_function",
}
EXP_CODES = (
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
    "HTP", "HDA", "HMP", "HGI", "HEP", "TAS", "IC",
)
NONEXP_CODES = (
    "IBA", "IEA", "IGC", "IKR", "ISA", "ISM", "ISO", "ISS", "NAS", "RCA",
)
NOT_RE = r"(^|\|)NOT(\||$)"
DUCKDB_THREADS = 8


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _parse_entry_id(header: str) -> str:
    parts = header.strip().split("|")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    first = header.strip().split()[0]
    if first:
        return first
    raise ValueError(f"Unexpected FASTA header: {header!r}")


def _iter_fasta_ids(path: Path) -> Iterator[str]:
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                yield _parse_entry_id(line[1:].strip())


def update_uniprot_annotations(spec: DatasetSpec) -> None:
    """Build current-snapshot GOA features for current train and query proteins."""
    if not spec.uniprot_gaf_parquet.exists():
        raise FileNotFoundError(f"UniProt-GOA parquet not found: {spec.uniprot_gaf_parquet}")

    spec.prepared_dir.mkdir(parents=True, exist_ok=True)
    terms_df = load_terms_df(spec.train_terms)
    train_ids_df = pd.DataFrame({"EntryID": terms_df["EntryID"].astype(str).unique()})
    test_ids_df = pd.DataFrame({
        "EntryID": pd.Index(list(_iter_fasta_ids(spec.test_fasta)), dtype="object").unique()
    })

    namespaces = list(ASPECT_TO_NS.values())
    snapshot = parse_obo_snapshot(str(spec.ontology_obo), namespaces)
    valid_go_ids = set()
    for namespace in namespaces:
        valid_go_ids.update(snapshot.edges_by_ns[namespace].keys())

    valid_terms_df = pd.DataFrame({"term": list(valid_go_ids)})
    alt_map_df = pd.DataFrame(
        {"alt": list(snapshot.alt_to_canon.keys()), "canon": list(snapshot.alt_to_canon.values())}
    )

    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"PRAGMA threads={int(DUCKDB_THREADS)}")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.register("valid_terms_df", valid_terms_df)
        con.register("alt_map_df", alt_map_df)
        con.register("test_ids_df", test_ids_df)
        con.register("train_ids_df", train_ids_df)
        con.execute("CREATE TEMP TABLE valid_terms AS SELECT * FROM valid_terms_df")
        con.execute("CREATE TEMP TABLE alt_map AS SELECT * FROM alt_map_df")
        con.execute("CREATE TEMP TABLE test_ids AS SELECT * FROM test_ids_df")
        con.execute("CREATE TEMP TABLE train_ids AS SELECT * FROM train_ids_df")
        con.execute(
            "CREATE TEMP TABLE allowed_ids AS "
            "SELECT EntryID FROM train_ids UNION SELECT EntryID FROM test_ids"
        )

        exp_list = ", ".join(_sql_str(code) for code in EXP_CODES)
        nonexp_list = ", ".join(_sql_str(code) for code in NONEXP_CODES)
        uniprot_path_sql = _sql_str(str(spec.uniprot_gaf_parquet))
        base_select_sql = f"""
            SELECT
                u.EntryID AS EntryID,
                COALESCE(m.canon, u.term) AS term,
                CASE u.aspect
                    WHEN 'P' THEN 'BPO'
                    WHEN 'F' THEN 'MFO'
                    WHEN 'C' THEN 'CCO'
                    ELSE u.aspect
                END AS aspect,
                u.taxon AS taxon,
                COALESCE(u.Qualifier, '') AS Qualifier,
                u.ECO AS ECO
            FROM read_parquet({uniprot_path_sql}) AS u
            LEFT JOIN alt_map AS m ON u.term = m.alt
            WHERE COALESCE(m.canon, u.term) IN (SELECT term FROM valid_terms)
        """

        con.execute(
            f"""
            COPY (
                WITH base AS ({base_select_sql})
                SELECT EntryID, term, aspect, taxon
                FROM base
                WHERE EntryID IN (SELECT EntryID FROM allowed_ids)
                  AND regexp_matches(Qualifier, '{NOT_RE}')
                  AND ECO IN ({exp_list})
                GROUP BY EntryID, term, aspect, taxon
            ) TO $1 (FORMAT 'parquet')
            """,
            [str(spec.not_terms_uniprot)],
        )

        con.execute(
            f"""
            COPY (
                WITH base AS (
                    {base_select_sql}
                    AND u.EntryID IN (SELECT EntryID FROM test_ids)
                )
                SELECT
                    EntryID, term, aspect, taxon,
                    MAX(CASE WHEN ECO IN ({exp_list}) THEN 1 ELSE 0 END) AS is_exp_confirmed
                FROM base
                GROUP BY EntryID, term, aspect, taxon
                HAVING MAX(CASE WHEN regexp_matches(Qualifier, '{NOT_RE}') THEN 1 ELSE 0 END) = 0
            ) TO $1 (FORMAT 'parquet')
            """,
            [str(spec.test_terms_uniprot)],
        )

        column_specs = [(code, f"nonexp_{code}") for code in NONEXP_CODES]
        nonexp_cols_sql = ",\n".join(
            "MAX(CASE WHEN ECO = " + _sql_str(code) + " THEN 1 ELSE 0 END) AS " + f'"{col}"'
            for code, col in column_specs
        )
        output_cols_sql = ", ".join(f'"{col}"' for _, col in column_specs)

        con.execute(
            f"""
            COPY (
                WITH base AS (
                    {base_select_sql}
                    AND u.EntryID IN (SELECT EntryID FROM allowed_ids)
                ),
                agg AS (
                    SELECT
                        EntryID, term, aspect, taxon,
                        MAX(CASE WHEN regexp_matches(Qualifier, '{NOT_RE}') THEN 1 ELSE 0 END) AS has_not,
                        {nonexp_cols_sql},
                        MAX(CASE WHEN ECO IN ({nonexp_list}) THEN 1 ELSE 0 END) AS has_any_nonexp
                    FROM base
                    GROUP BY EntryID, term, aspect, taxon
                )
                SELECT EntryID, term, aspect, taxon, {output_cols_sql}
                FROM agg
                WHERE has_not = 0 AND has_any_nonexp = 1
            ) TO $1 (FORMAT 'parquet')
            """,
            [str(spec.nonexp_codes_uniprot)],
        )
    finally:
        con.close()

    print(f"[uniprot] wrote: {spec.not_terms_uniprot}")
    print(f"[uniprot] wrote: {spec.test_terms_uniprot}")
    print(f"[uniprot] wrote: {spec.nonexp_codes_uniprot}")
