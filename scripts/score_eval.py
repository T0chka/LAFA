import argparse

from src.evaluation.window_score import score_window


parser = argparse.ArgumentParser()
parser.add_argument("--groundtruth-dir", required=True)
parser.add_argument("--source-work", required=True)
parser.add_argument("--source-snapshot", required=True)
parser.add_argument("--future-snapshot", required=True)
parser.add_argument("--output")
parser.add_argument("--th-step", type=float, default=0.01)
args = parser.parse_args()

score_window(
    args.groundtruth_dir,
    args.source_work,
    args.source_snapshot,
    args.future_snapshot,
    args.output,
    args.th_step,
)
