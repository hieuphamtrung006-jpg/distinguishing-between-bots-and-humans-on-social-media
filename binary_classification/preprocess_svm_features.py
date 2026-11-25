"""Tiền xử lý và trích xuất 18 đặc trưng cho nhánh SVM (BotSim-24 & Arctic Shift).
"""

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Các đặc trưng bị chặn biên trong [0, 1] hoặc [-1, 1], không áp dụng Yeo-Johnson.
# Lưu ý: sub_to_com_ratio KHÔNG nằm trong danh sách này vì không có cận trên (skew rất lớn).
BOUNDED_EXCLUDE_COLS = {
    "comment_1_ratio",
    "comment_2_ratio",
    "upvote_ratio_mean",
    "upvote_ratio_std",
    "burstiness",
}

ZERO_IMPUTE_COLS = ["dt_std", "score_std", "upvote_ratio_std", "circadian_entropy"]
MEDIAN_IMPUTE_COLS = [
    "dt_mean",
    "dt_median",
    "dt_min",
    "dt_max",
    "score_mean",
    "upvote_ratio_mean",
    "burstiness",
]

EPOCH_UTC = pd.Timestamp("1970-01-01", tz="UTC")


def assign_labels(users: pd.DataFrame, botsim_bot_count: int = 1000) -> pd.DataFrame:
    """Gán nhãn chuẩn hóa: 0 = Bot, 1 = Human/User.

    Giữ nguyên cột 'label' nếu dữ liệu đầu vào (ví dụ Arctic Shift) đã có sẵn;
    ngược lại gán nhãn dựa trên 'character_setting' hoặc botsim_bot_count dòng cuối.
    """
    df = users.copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
        return df

    df["label"] = 1
    if "character_setting" in df.columns and df["character_setting"].notna().any():
        df.loc[df["character_setting"].notna(), "label"] = 0
    elif len(df) >= botsim_bot_count > 0:
        df.loc[df.index[-botsim_bot_count:], "label"] = 0
    return df


def _extract_temporal_and_circadian(posts: List[Dict[str, Any]]) -> Tuple[float, ...]:
    """Trích xuất các đặc trưng khoảng trễ (dt), burstiness và nhịp sinh học 24h."""
    raw_times = [p.get("created_utc") for p in posts if p.get("created_utc")]
    if not raw_times:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    dates = (
        pd.Series(pd.to_datetime(raw_times, errors="coerce", utc=True))
        .dropna()
        .sort_values()
    )
    if dates.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # Tính số giây Unix timestamp an toàn, không phụ thuộc độ phân giải ns/us của pandas
    timestamps_sec = (dates - EPOCH_UTC).dt.total_seconds().to_numpy()
    dt = np.diff(timestamps_sec)

    if len(dt) > 0:
        dt_mean = float(np.mean(dt))
        dt_median = float(np.median(dt))
        dt_min = float(np.min(dt))
        dt_max = float(np.max(dt))
    else:
        dt_mean = dt_median = dt_min = dt_max = np.nan

    dt_std = float(np.std(dt, ddof=1)) if len(dt) > 1 else np.nan

    if len(dt) > 1:
        denom = dt_std + dt_mean
        # Nếu dt_std == 0 và dt_mean == 0 (đăng hàng loạt trong 0s), giữ -1.0 thay vì NaN
        burstiness = float((dt_std - dt_mean) / denom) if denom > 0 else -1.0
    else:
        burstiness = np.nan

    hours = dates.dt.hour.to_numpy()
    counts = np.bincount(hours, minlength=24)
    probs = counts[counts > 0] / len(hours)
    circadian_entropy = float(-np.sum(probs * np.log(probs)))

    return (
        dt_mean,
        dt_std,
        dt_median,
        dt_min,
        dt_max,
        burstiness,
        circadian_entropy,
    )


def extract_user_features(valid_users: pd.DataFrame, activity_data: Dict[str, Any]) -> pd.DataFrame:
    """Trích xuất 18 đặc trưng thô và 11 cột chỉ báo khuyết thiếu cho từng user."""
    records: List[Dict[str, Any]] = []

    for _, user in valid_users.iterrows():
        uid = user["user_id"]
        info = activity_data.get(uid, {})

        posts = info.get("posts") or []
        comments1 = info.get("comment_1") or []
        comments2 = info.get("comment_2") or []

        # Nhóm 1: Hoạt động người dùng
        submission_num = float(user.get("submission_num") or 0)
        comment_num = float(user.get("comment_num") or 0)
        c1 = float(user.get("comment_num_1") or 0)
        c2 = float(user.get("comment_num_2") or 0)

        comment_1_ratio = c1 / (comment_num + 1.0)
        comment_2_ratio = c2 / (comment_num + 1.0)
        sub_to_com_ratio = submission_num / (comment_num + 1.0)

        desc = user.get("description")
        description_len = len(str(desc)) if pd.notna(desc) else 0

        # Nhóm 2 & 3: Khoảng trễ hoạt động và nhịp sinh học 24h
        (
            dt_mean,
            dt_std,
            dt_median,
            dt_min,
            dt_max,
            burstiness,
            circadian_entropy,
        ) = _extract_temporal_and_circadian(posts)

        # Nhóm 4: Tương tác & cộng đồng
        scores = [p.get("score") for p in posts if p.get("score") is not None]
        upvotes = [p.get("upvote_ratio") for p in posts if p.get("upvote_ratio") is not None]

        score_mean = float(np.mean(scores)) if scores else np.nan
        score_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan
        upvote_ratio_mean = float(np.mean(upvotes)) if upvotes else np.nan
        upvote_ratio_std = float(np.std(upvotes, ddof=1)) if len(upvotes) > 1 else np.nan

        subs = {
            item["subreddit"]
            for item in (posts + comments1 + comments2)
            if item.get("subreddit")
        }

        records.append(
            {
                "user_id": uid,
                "label": int(user["label"]),
                "submission_num": submission_num,
                "comment_num": comment_num,
                "comment_1_ratio": comment_1_ratio,
                "comment_2_ratio": comment_2_ratio,
                "sub_to_com_ratio": sub_to_com_ratio,
                "description_len": description_len,
                "dt_mean": dt_mean,
                "dt_std": dt_std,
                "dt_median": dt_median,
                "dt_min": dt_min,
                "dt_max": dt_max,
                "burstiness": burstiness,
                "circadian_entropy": circadian_entropy,
                "score_mean": score_mean,
                "score_std": score_std,
                "upvote_ratio_mean": upvote_ratio_mean,
                "upvote_ratio_std": upvote_ratio_std,
                "unique_subreddit_num": len(subs),
            }
        )

    df = pd.DataFrame(records)
    indicator_cols = ZERO_IMPUTE_COLS + MEDIAN_IMPUTE_COLS
    missing_ind = df[indicator_cols].isna().astype(int)
    missing_ind.columns = [f"{col}_missing" for col in indicator_cols]
    return pd.concat([df, missing_ind], axis=1)


def split_dataset(
    df: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu phân tầng 70% Train / 20% Validation / 10% Test."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1.0 / 3.0, stratify=temp_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def fit_and_transform_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Điền khuyết, biến đổi Yeo-Johnson và chuẩn hóa StandardScaler chỉ fit trên Train."""
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    # 1. Điền 0 cho các cột đo biến thiên / entropy
    for split in (train_out, val_out, test_out):
        split[ZERO_IMPUTE_COLS] = split[ZERO_IMPUTE_COLS].fillna(0.0)

    # 2. Điền median tính ĐỘC LẬP trên tập Train
    train_medians = {col: float(train_out[col].median()) for col in MEDIAN_IMPUTE_COLS}
    for col, med_val in train_medians.items():
        fill_val = 0.0 if pd.isna(med_val) else med_val
        for split in (train_out, val_out, test_out):
            split[col] = split[col].fillna(fill_val)

    # 3. Kiểm tra độ lệch (skewness) trên Train và áp dụng Yeo-Johnson
    num_cols = [
        c
        for c in train_out.columns
        if c not in ("user_id", "label") and not c.endswith("_missing")
    ]
    yeo_cols = [
        col
        for col in num_cols
        if col not in BOUNDED_EXCLUDE_COLS
        and float(train_out[col].std(ddof=0)) > 1e-8
        and abs(float(stats.skew(train_out[col].dropna()))) > 1.0
    ]

    power_transformer = None
    if yeo_cols:
        power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        train_out[yeo_cols] = power_transformer.fit_transform(train_out[yeo_cols])
        val_out[yeo_cols] = power_transformer.transform(val_out[yeo_cols])
        test_out[yeo_cols] = power_transformer.transform(test_out[yeo_cols])

    # 4. Chuẩn hóa StandardScaler cho toàn bộ đặc trưng số (chỉ fit trên Train)
    scaler = StandardScaler()
    train_out[num_cols] = scaler.fit_transform(train_out[num_cols])
    val_out[num_cols] = scaler.transform(val_out[num_cols])
    test_out[num_cols] = scaler.transform(test_out[num_cols])

    artifacts = {
        "train_medians": train_medians,
        "yeo_cols": yeo_cols,
        "num_cols": num_cols,
        "power_transformer": power_transformer,
        "scaler": scaler,
    }
    return train_out, val_out, test_out, artifacts


def run_pipeline(
    users_path: str,
    posts_path: str,
    out_dir: str,
    seed: int = 42,
    botsim_bot_count: int = 1000,
) -> None:
    """Thực thi toàn bộ pipeline tiền xử lý không rò rỉ dữ liệu."""
    os.makedirs(out_dir, exist_ok=True)

    users = pd.read_csv(users_path)
    with open(posts_path, "r", encoding="utf-8") as f:
        activity_data = json.load(f)

    users = assign_labels(users, botsim_bot_count=botsim_bot_count)
    valid_users = users[users["user_id"].isin(set(activity_data.keys()))].copy()

    raw_df = extract_user_features(valid_users, activity_data)
    raw_path = os.path.join(out_dir, "dataset_18_features_raw.csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    train_raw, val_raw, test_raw = split_dataset(raw_df, seed=seed)
    train_df, val_df, test_df, artifacts = fit_and_transform_splits(
        train_raw, val_raw, test_raw
    )

    train_df.to_csv(os.path.join(out_dir, "train_features.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(out_dir, "val_features.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(out_dir, "test_features.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "preprocessors.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    print(f"[*] User hợp lệ: {len(raw_df)} | Phân bố nhãn: {raw_df['label'].value_counts().to_dict()}")
    print(f"[*] Chia tập (Train/Val/Test): {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"[*] Các cột áp dụng Yeo-Johnson (fit trên Train): {artifacts['yeo_cols']}")
    print(f"[*] Đã lưu dữ liệu sạch vào thư mục: {out_dir}")


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Trích xuất và chuẩn hóa 18 đặc trưng SVM không rò rỉ dữ liệu."
    )
    parser.add_argument("--users", default="Users.csv", help="Đường dẫn tới Users.csv")
    parser.add_argument("--posts", default="user_post_comment.json", help="Đường dẫn tới JSON hoạt động")
    parser.add_argument("--out-dir", default="outputs", help="Thư mục lưu đầu ra")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cho chia tập")
    parser.add_argument(
        "--botsim-bot-count",
        type=int,
        default=1000,
        help="Số lượng bot ở cuối tệp nếu dữ liệu chưa có cột label/character_setting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        users_path=args.users,
        posts_path=args.posts,
        out_dir=args.out_dir,
        seed=args.seed,
        botsim_bot_count=args.botsim_bot_count,
    )
