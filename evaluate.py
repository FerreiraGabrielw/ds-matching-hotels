#!/usr/bin/env python3
import argparse
import json
import time
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predictions using train.csv as ground-truth (is_match==1). "
            "Predictions not present in train.csv pairs are ignored to avoid label-missing false positives."
        )
    )
    parser.add_argument(
        "--pred",
        required=True,
        help=(
            "Path to predictions CSV with columns: id_A,id_B,predicted_match (0/1)."
        ),
    )
    parser.add_argument(
        "--truth",
        default="data/train.csv",
        help=(
            "Path to training pairs CSV (id_A,id_B,is_match). Only is_match==1 is used as positives."
        ),
    )
    parser.add_argument(
        "--out",
        default="metrics.json",
        help="Where to save metrics JSON (precision/recall/F1).",
    )
    args = parser.parse_args()

    t0 = time.time()
    pred = load_csv(args.pred)
    truth = load_csv(args.truth)

    # Validate schema
    required_pred_cols = {"id_A", "id_B", "predicted_match"}
    if not required_pred_cols.issubset(set(pred.columns)):
        missing = required_pred_cols - set(pred.columns)
        raise ValueError(f"Missing columns in predictions: {missing}")

    required_truth_cols = {"id_A", "id_B", "is_match"}
    if not required_truth_cols.issubset(set(truth.columns)):
        missing = required_truth_cols - set(truth.columns)
        raise ValueError(f"Missing columns in truth/train: {missing}")

    # Consider only predicted positives that exist in the training pairs universe
    train_pairs_universe = truth[["id_A", "id_B"]].drop_duplicates()
    pred_pos_all = pred[pred["predicted_match"] == 1][["id_A", "id_B"]].drop_duplicates()
    pred_pos = pd.merge(pred_pos_all, train_pairs_universe, on=["id_A", "id_B"], how="inner")
    truth_pos = truth[truth["is_match"] == 1][["id_A", "id_B"]].drop_duplicates()

    # True positives: exact pair intersection
    merged = pd.merge(pred_pos, truth_pos, on=["id_A", "id_B"], how="inner")
    tp = int(len(merged))
    fp = int(len(pred_pos) - tp)
    fn = int(len(truth_pos) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "eval_time_sec": round(time.time() - t0, 3),
    }

    print(json.dumps(metrics, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()


