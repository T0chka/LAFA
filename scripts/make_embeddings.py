from config import dataset
from src.core.debug import print_separator
from src.embeddings.generate import generate_embeddings
from src.embeddings.specs import ESM1B_650M, ESM2_3B, PROT_T5


if __name__ == "__main__":
    ds = dataset()
    print_separator("make_embeddings", "Generate or reuse protein embeddings", char="=")
    for spec in (ESM2_3B, PROT_T5, ESM1B_650M):
        generate_embeddings(ds, spec, log_prefix="make_embeddings")
