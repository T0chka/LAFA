from config import dataset
from src.core.debug import print_separator
from src.models.predictor import PredictorSpec, train_predictor


SPEC = PredictorSpec(
    name="hmlp_esm2",
    model="hmlp",
    embeddings=("esm2_t36_3B_UR50D",),
    min_freq={"BPO": 10, "CCO": 0, "MFO": 0},
)


if __name__ == "__main__":
    print_separator("build_hmlp", "Train HMLP", char="=")
    train_predictor(dataset(), SPEC, log_prefix="build_hmlp")
