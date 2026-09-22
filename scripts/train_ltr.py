from config import dataset
from src.core.debug import print_separator
from src.ltr.ranker import train_ltr


if __name__ == "__main__":
    print_separator("train_ltr", "Train learning-to-rank ensemble", char="=")
    train_ltr(dataset(), log_prefix="train_ltr")
