from config import dataset
from src.core.debug import print_separator
from src.models.blast_knn import build_blast_oof_component
from src.models.blast_search import ensure_blast_train_hits


if __name__ == "__main__":
    ds = dataset()
    print_separator("build_blast", "Build full BLAST-KNN OOF component", char="=")
    train_hits = ensure_blast_train_hits(ds, log_prefix="build_blast")
    build_blast_oof_component(ds, train_hits_path=train_hits, log_prefix="build_blast")
