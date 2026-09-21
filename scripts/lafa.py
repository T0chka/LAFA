"""Run the unchanged CAFA6 ensemble on one LAFA snapshot."""

import argparse
import gzip
import shutil
from pathlib import Path

from config import dataset_from_paths
from scripts.build_hmlp import SPEC as HMLP_SPEC
from scripts.build_mlp import SPEC as MLP_SPEC
from scripts.build_pyboost import SPEC as PYBOOST_SPEC
from src.data.prepare import prepare_dataset
from src.embeddings.generate import generate_embeddings
from src.embeddings.specs import ESM1B_650M, ESM2_3B, PROT_T5
from src.ltr.ranker import predict_ltr, train_ltr
from src.models.blast_knn import build_blast_component
from src.models.blast_search import ensure_blast_hits
from src.models.naive_prior import build_naive_component
from src.models.nonexp import build_nonexp_component
from src.models.predictor import build_predictor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the protein-function ensemble on a LAFA snapshot.")
    parser.add_argument("--query_file", "-q", required=True)
    parser.add_argument("--train_sequences", required=True)
    parser.add_argument("--annot_file", "-a", required=True, help="Current train_terms.tsv")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--goa_gaf", required=True)
    parser.add_argument("--output_file", "-o", required=True)
    parser.add_argument("--work_dir", default="artifacts/run")
    parser.add_argument("--embedding_cache", default="artifacts/cache/embeddings")
    parser.add_argument("--ia_file", default=None)
    parser.add_argument("--num_threads", type=int, default=None)
    return parser


def _copy_output(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".gz":
        with source.open("rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copyfile(source, target)


def main() -> None:
    args = _parser().parse_args()
    ds = dataset_from_paths(
        train_sequences=args.train_sequences,
        query_file=args.query_file,
        train_terms=args.annot_file,
        graph=args.graph,
        goa_gaf=args.goa_gaf,
        work_dir=args.work_dir,
        embedding_cache=args.embedding_cache,
        ia_file=args.ia_file,
    )

    prepare_dataset(ds)
    for spec in (ESM2_3B, PROT_T5, ESM1B_650M):
        generate_embeddings(ds, spec)
    build_predictor(ds, HMLP_SPEC)
    build_predictor(ds, MLP_SPEC)
    build_predictor(ds, PYBOOST_SPEC)
    train_hits, test_hits = ensure_blast_hits(ds, threads=args.num_threads)
    build_blast_component(ds, train_hits, test_hits)
    build_naive_component(ds)
    build_nonexp_component(ds)
    train_ltr(ds)
    predict_ltr(ds)

    source = Path(args.work_dir) / "final" / "submission.tsv"
    _copy_output(source, Path(args.output_file))
    print(f"[lafa] wrote: {args.output_file}")


if __name__ == "__main__":
    main()
