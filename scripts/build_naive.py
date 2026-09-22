from config import dataset
from src.core.debug import print_separator
from src.models.naive_prior import build_naive_oof


if __name__ == "__main__":
    print_separator("build_naive", "Build naive-prior OOF component", char="=")
    build_naive_oof(dataset(), log_prefix="build_naive")
