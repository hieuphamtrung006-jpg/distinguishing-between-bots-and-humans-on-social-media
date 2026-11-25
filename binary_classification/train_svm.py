import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from preprocess_svm_features import fit_and_transform_splits, split_dataset

# Cac nhom dac trung de doi chung anh huong cua dac trung mo phong trong BotSim-24
FEATURE_GROUPS = {
    "full_18": [],  # Du 18 dac trung (+11 co missing = 29 cot)
    "no_engagement": ["score_", "upvote_ratio_"],  # Bo score & upvote_ratio (bot mo phong deu = 0)
    "behavioral_only": ["score_", "upvote_ratio_", "description_len"],  # Chi hanh vi & thoi gian
}


def eval_predictions(model, X, y_true):
    # Quy uoc: 0 = Bot, 1 = Human -> tinh Precision/Recall/F1 cho lop Bot (0)
    bot_idx = list(model.classes_).index(0)
    prob_bot = model.predict_proba(X)[:, bot_idx]
    y_pred = model.predict(X)

    y_bot_true = (y_true == 0).astype(int)
    y_bot_pred = (y_pred == 0).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_bot": float(precision_score(y_bot_true, y_bot_pred, zero_division=0)),
        "recall_bot": float(recall_score(y_bot_true, y_bot_pred, zero_division=0)),
        "f1_bot": float(f1_score(y_bot_true, y_bot_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_bot_true, prob_bot)),
    }


def train_and_select_svm(X_train, y_train, X_val, y_val, kernel, c_grid, seed):
    best_model = None
    best_c = None
    best_val_f1 = -1.0

    for c_val in c_grid:
        clf = SVC(
            C=c_val,
            kernel=kernel,
            probability=True,  # Platt Scaling
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        val_f1 = eval_predictions(clf, X_val, y_val)["f1_bot"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c_val
            best_model = clf

    return best_model, best_c


def run_split_for_group(train_df, val_df, test_df, drop_prefixes, c_grid, seed, verbose=False):
    feat_cols = [
        c
        for c in train_df.columns
        if c not in ("user_id", "label") and not any(c.startswith(p) for p in drop_prefixes)
    ]
    X_train, y_train = train_df[feat_cols].to_numpy(), train_df["label"].to_numpy()
    X_val, y_val = val_df[feat_cols].to_numpy(), val_df["label"].to_numpy()
    X_test, y_test = test_df[feat_cols].to_numpy(), test_df["label"].to_numpy()

    res = {}
    for kernel in ("linear", "rbf"):
        model, best_c = train_and_select_svm(X_train, y_train, X_val, y_val, kernel, c_grid, seed)
        metrics = eval_predictions(model, X_test, y_test)
        metrics["best_C"] = best_c
        metrics["num_features"] = len(feat_cols)
        res[kernel] = metrics

        if verbose:
            y_pred = model.predict(X_test)
            print(f"\n[Seed {seed}] SVM-{kernel} (best_C={best_c}, n_feats={len(feat_cols)})")
            print("Confusion matrix [Bot=0, Human=1]:")
            print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
            print(classification_report(y_test, y_pred, target_names=["Bot (0)", "Human (1)"], digits=4))

    return res


def summarize_group(group_name, runs):
    metrics = ["accuracy", "precision_bot", "recall_bot", "f1_bot", "macro_f1", "roc_auc"]
    summary = {}
    print(f"\n=== Tong hop {len(runs)} seeds [{group_name}] ===")
    for kernel in ("linear", "rbf"):
        summary[kernel] = {}
        n_feats = runs[0][kernel]["num_features"]
        print(f"SVM ({kernel.upper()}, {n_feats} cols):")
        for m in metrics:
            vals = [r[kernel][m] for r in runs]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary[kernel][m] = {"mean": mean_v, "std": std_v}
            print(f"  {m:15s}: {mean_v:.4f} +/- {std_v:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Huan luyen va danh gia SVM Baseline tren BotSim-24.")
    parser.add_argument("--data-dir", default="outputs", help="Thu muc chua du lieu da tien xu ly")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["full_18", "no_engagement", "behavioral_only"],
        choices=list(FEATURE_GROUPS.keys()),
    )
    args = parser.parse_args()

    raw_path = os.path.join(args.data_dir, "dataset_18_features_raw.csv")
    use_raw = os.path.exists(raw_path)
    if use_raw:
        raw_df = pd.read_csv(raw_path)
    else:
        tr_fixed = pd.read_csv(os.path.join(args.data_dir, "train_features.csv"))
        va_fixed = pd.read_csv(os.path.join(args.data_dir, "val_features.csv"))
        te_fixed = pd.read_csv(os.path.join(args.data_dir, "test_features.csv"))

    all_summaries = {}
    all_runs = {}

    for group_name in args.groups:
        drop_prefixes = FEATURE_GROUPS[group_name]
        runs = []
        if use_raw:
            for idx, seed in enumerate(args.seeds):
                tr_r, va_r, te_r = split_dataset(raw_df, seed=seed)
                tr_d, va_d, te_d, _ = fit_and_transform_splits(tr_r, va_r, te_r)
                show_detail = idx == 0 and group_name == "behavioral_only"
                runs.append(
                    run_split_for_group(
                        tr_d, va_d, te_d, drop_prefixes, args.c_grid, seed, verbose=show_detail
                    )
                )
        else:
            runs.append(
                run_split_for_group(
                    tr_fixed,
                    va_fixed,
                    te_fixed,
                    drop_prefixes,
                    args.c_grid,
                    args.seeds[0],
                    verbose=(group_name == "behavioral_only"),
                )
            )
        all_runs[group_name] = runs
        all_summaries[group_name] = summarize_group(group_name, runs)

    out_path = os.path.join(args.data_dir, "svm_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"seeds": args.seeds, "summary": all_summaries, "runs": all_runs}, f, indent=2)
    print(f"\nDa luu ket qua chi tiet tai: {out_path}")


if __name__ == "__main__":
    main()
