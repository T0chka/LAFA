from config import dataset
from src.core.debug import print_separator
from src.data.prepare import prepare_dataset


if __name__ == "__main__":
    print_separator("prepare_data", "Prepare snapshot data", char="=")
    prepare_dataset(dataset(), log_prefix="prepare_data")
