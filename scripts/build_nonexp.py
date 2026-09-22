from config import dataset
from src.core.debug import print_separator
from src.models.nonexp import build_nonexp_oof


if __name__ == "__main__":
    print_separator("build_nonexp", "Build non-experimental OOF component", char="=")
    build_nonexp_oof(dataset(), log_prefix="build_nonexp")
