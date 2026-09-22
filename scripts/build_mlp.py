from config import dataset
from src.core.debug import print_separator
from src.models.predictor import PredictorSpec, train_predictor


SPEC = PredictorSpec(
    name="mlp_t5_esm1b",
    model="mlp",
    embeddings=("prot_t5", "esm1b_650M"),
    min_freq={"BPO": 10, "CCO": 0, "MFO": 0},
)


if __name__ == "__main__":
    print_separator("build_mlp", "Train MLP", char="=")
    train_predictor(dataset(), SPEC, log_prefix="build_mlp")
