from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    train_fasta: Path
    test_fasta: Path
    train_terms: Path
    ontology_obo: Path
    ia_file: Path
    prepared_dir: Path
    cache_dir: Path
    embedding_cache_dir: Path
    uniprot_gaf_parquet: Path
    uniprot_gaf_gz: Path | None = None
    use_uniprot: bool = True

    @property
    def test_terms_uniprot(self) -> Path:
        return self.prepared_dir / "uniprot_terms_test_prots.parquet"

    @property
    def not_terms_uniprot(self) -> Path:
        return self.prepared_dir / "uniprot_not_qualifier.parquet"

    @property
    def nonexp_codes_uniprot(self) -> Path:
        return self.prepared_dir / "uniprot_nonexp_ohe.parquet"

    @property
    def train_index(self) -> Path:
        return self.prepared_dir / "train_index.parquet"

    @property
    def test_index(self) -> Path:
        return self.prepared_dir / "test_index.parquet"

    @property
    def ground_truth(self) -> Path:
        return self.prepared_dir / "ground_truth.pkl"
