import os
from pathlib import Path

from src.data.dataset import DatasetSpec


def dataset_from_paths(
    train_sequences: str | Path,
    query_file: str | Path,
    train_terms: str | Path,
    graph: str | Path,
    goa_gaf: str | Path,
    work_dir: str | Path = "artifacts",
    embedding_cache: str | Path = "artifacts/cache/embeddings",
    ia_file: str | Path | None = None,
) -> DatasetSpec:
    work_dir = Path(work_dir)
    prepared_dir = work_dir / "prepared"
    cache_dir = work_dir / "cache"
    return DatasetSpec(
        train_fasta=Path(train_sequences),
        test_fasta=Path(query_file),
        train_terms=Path(train_terms),
        ontology_obo=Path(graph),
        ia_file=Path(ia_file) if ia_file is not None else prepared_dir / "information_accretion.tsv",
        prepared_dir=prepared_dir,
        cache_dir=cache_dir,
        embedding_cache_dir=Path(embedding_cache),
        uniprot_gaf_parquet=cache_dir / "goa_uniprot_sprot.parquet",
        uniprot_gaf_gz=Path(goa_gaf),
        use_uniprot=True,
    )


def dataset() -> DatasetSpec:
    required = {
        "LAFA_TRAIN_SEQUENCES": os.environ.get("LAFA_TRAIN_SEQUENCES"),
        "LAFA_QUERY_FILE": os.environ.get("LAFA_QUERY_FILE"),
        "LAFA_TRAIN_TERMS": os.environ.get("LAFA_TRAIN_TERMS"),
        "LAFA_GRAPH": os.environ.get("LAFA_GRAPH"),
        "LAFA_GOA_GAF": os.environ.get("LAFA_GOA_GAF"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("Missing LAFA dataset environment variables: " + ", ".join(missing))
    return dataset_from_paths(
        train_sequences=required["LAFA_TRAIN_SEQUENCES"],
        query_file=required["LAFA_QUERY_FILE"],
        train_terms=required["LAFA_TRAIN_TERMS"],
        graph=required["LAFA_GRAPH"],
        goa_gaf=required["LAFA_GOA_GAF"],
        work_dir=os.environ.get("LAFA_WORK_DIR", "artifacts"),
        embedding_cache=os.environ.get("LAFA_EMBEDDING_CACHE", "artifacts/cache/embeddings"),
        ia_file=os.environ.get("LAFA_IA_FILE"),
    )
