from config import dataset
from src.core.debug import print_separator
from src.models.predictor import PredictorSpec, train_predictor


SPEC = PredictorSpec(
    name="pyb_t5",
    model="pyboost",
    embeddings=("prot_t5",),
    min_freq={"BPO": 10, "CCO": 0, "MFO": 0},
)


if __name__ == "__main__":
    print_separator("build_pyboost", "Train PyBoost", char="=")
    train_predictor(dataset(), SPEC, log_prefix="build_pyboost")
