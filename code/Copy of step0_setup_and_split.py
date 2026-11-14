"""
Bước 0 (Giai đoạn 0): Chuẩn bị dữ liệu, Hạ kiểu RAM có kiểm soát, Định nghĩa Schema 432 biến
và Chốt ranh giới chia tập (Timestamp-Safe Temporal Split & Stratified Split).

Thiết kế cho Phần 2 (Người 2 - Advanced EDA, Feature Engineering & Feature Selection)
Dự án: IEEE-CIS Fraud Detection (Midterm Report)
"""

from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ==============================================================================
# 1. TIỆN ÍCH ĐO BỘ NHỚ TIẾN TRÌNH (PROCESS RSS / PEAK RSS & DATAFRAME MEMORY)
# ==============================================================================
def get_process_rss_mb() -> Dict[str, float]:
    """
    Đo lượng RAM thực tế của tiến trình (Current RSS và Peak RSS tính bằng MB).
    Tương thích mọi môi trường (dùng psutil nếu có, hoặc đọc trực tiếp từ /proc/self/status & resource).
    """
    current_rss_mb = 0.0
    peak_rss_mb = 0.0

    try:
        import psutil  # type: ignore
        proc = psutil.Process(os.getpid())
        current_rss_mb = proc.memory_info().rss / (1024 ** 2)
    except ImportError:
        status_path = Path("/proc/self/status")
        if status_path.exists():
            for line in status_path.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    current_rss_mb = float(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    peak_rss_mb = float(line.split()[1]) / 1024.0

    ru_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = (1024.0 ** 2) if sys.platform == "darwin" else 1024.0
    peak_rss_mb = max(peak_rss_mb, float(ru_max) / divisor, current_rss_mb)

    return {
        "current_rss_mb": round(current_rss_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }


def get_df_memory_mb(df: pd.DataFrame) -> float:
    """Trả về dung lượng bộ nhớ sâu (deep=True) của DataFrame tính bằng MB."""
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


# ==============================================================================
# 2. ĐỊNH NGHĨA CHÍNH XÁC SCHEMA 432 BIẾN GỐC (383 NUMERICAL + 49 CATEGORICAL)
# ==============================================================================
def get_ieee_cis_canonical_schema() -> Tuple[List[str], List[str]]:
    """
    Khởi tạo danh sách chuẩn 383 biến số (Numerical) và 49 biến phân loại (Categorical)
    theo đúng đặc tả chính thức của cuộc thi IEEE-CIS Fraud Detection và Phần 1.
    """
    # 383 biến Numerical:
    # 2 (TransactionDT, TransactionAmt) + 2 (dist1, dist2) + 14 (C1-C14)
    # + 15 (D1-D15) + 339 (V1-V339) + 11 (id_01-id_11) = 383
    num_cols: List[str] = (
        ["TransactionDT", "TransactionAmt", "dist1", "dist2"]
        + [f"C{i}" for i in range(1, 15)]
        + [f"D{i}" for i in range(1, 16)]
        + [f"V{i}" for i in range(1, 340)]
        + [f"id_{i:02d}" for i in range(1, 12)]
    )

    # 49 biến Categorical (bao gồm cả các mã số thực chất là định danh nhóm như card1-6, addr1-2):
    # 1 (ProductCD) + 6 (card1-card6) + 2 (addr1, addr2) + 2 (P/R_emaildomain)
    # + 9 (M1-M9) + 27 (id_12-id_38) + 2 (DeviceType, DeviceInfo) = 49
    cat_cols: List[str] = (
        ["ProductCD"]
        + [f"card{i}" for i in range(1, 7)]
        + ["addr1", "addr2", "P_emaildomain", "R_emaildomain"]
        + [f"M{i}" for i in range(1, 10)]
        + [f"id_{i:02d}" for i in range(12, 39)]
        + ["DeviceType", "DeviceInfo"]
    )

    assert len(num_cols) == 383, f"Lỗi Schema: Số biến Numerical phải là 383, hiện tại là {len(num_cols)}"
    assert len(cat_cols) == 49, f"Lỗi Schema: Số biến Categorical phải là 49, hiện tại là {len(cat_cols)}"
    assert len(set(num_cols).intersection(set(cat_cols))) == 0, "Lỗi Schema: Có biến trùng giữa Numerical và Categorical"

    return num_cols, cat_cols


# ==============================================================================
# 3. TÁCH PHẦN XU LẺ (AMT_CENTS) TRÊN FLOAT64 & TẠO CỜ HAS_IDENTITY GỐC
# ==============================================================================
def extract_pre_downcast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện các phép trích xuất bắt buộc TRƯỚC KHI ép kiểu bộ nhớ (downcast):
    1. Chuẩn hóa tên cột id-NN -> id_NN (nếu đọc từ file gốc chưa đổi tên).
    2. Tạo cờ has_identity (0/1) trực tiếp từ sự hiện diện của 40 trường bảng Identity.
    3. Tách phần xu lẻ Amt_cents (0..99) từ TransactionAmt khi còn ở độ chính xác float64.
    """
    rename_map = {f"id-{i:02d}": f"id_{i:02d}" for i in range(1, 39) if f"id-{i:02d}" in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    identity_cols = [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"]
    present_id_cols = [c for c in identity_cols if c in df.columns]

    # Tạo cờ has_identity = 1 nếu dòng giao dịch có khớp bản ghi Identity (ít nhất 1 cột Identity khác NaN)
    df["has_identity"] = df[present_id_cols].notna().any(axis=1).astype("int8")

    # Tách phần xu lẻ (0..99) trên kiểu float64 gốc để tránh sai số dấu phẩy động của float32
    amt_f64 = df["TransactionAmt"].to_numpy(dtype="float64")
    cents_int64 = np.round(amt_f64 * 100.0).astype("int64")
    df["Amt_cents"] = (cents_int64 % 100).astype("int8")

    return df


# ==============================================================================
# 4. HẠ KIỂU BỘ NHỚ (MEMORY DOWNCASTING) CÓ KIỂM TRA MIỀN GIÁ TRỊ & SAI SỐ
# ==============================================================================
def downcast_dataframe_safely(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    max_rel_error_tol: float = 1e-6,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Hạ kiểu dữ liệu có kiểm soát:
    - Kiểm tra miền min/max trước khi hạ kiểu số nguyên (int64 -> int8/int16/int32).
    - Kiểm tra sai số tương đối trước khi hạ kiểu số thực (float64 -> float32).
    - Không ép các mã phân loại dạng số (card1-6, addr1-2) sang float32 nếu làm sai lệch mã nguyên.
    """
    mem_before_mb = get_df_memory_mb(df)
    rss_before = get_process_rss_mb()

    # Kiểm tra riêng sai số khi chuyển TransactionAmt từ float64 sang float32
    amt_f64 = df["TransactionAmt"].to_numpy(dtype="float64")
    amt_f32 = amt_f64.astype("float32")
    valid_mask = np.isfinite(amt_f64) & (amt_f64 > 0)
    max_amt_rel_err = float(np.max(np.abs(amt_f64[valid_mask] - amt_f32[valid_mask]) / amt_f64[valid_mask]))
    assert max_amt_rel_err < max_rel_error_tol, (
        f"Sai số tương đối của TransactionAmt khi sang float32 ({max_amt_rel_err:.2e}) vượt ngưỡng {max_rel_error_tol}"
    )

    # 1. Xử lý các cột khoá và nhãn (TransactionID, isFraud, TransactionDT)
    for int_col in ["TransactionID", "isFraud", "TransactionDT"]:
        if int_col in df.columns and pd.api.types.is_integer_dtype(df[int_col]):
            c_min, c_max = df[int_col].min(), df[int_col].max()
            for dtype_candidate in [np.int8, np.int16, np.int32, np.int64]:
                info = np.iinfo(dtype_candidate)
                if c_min >= info.min and c_max <= info.max:
                    df[int_col] = df[int_col].astype(dtype_candidate)
                    break

    # 2. Hạ kiểu cho 383 biến Numerical
    for col in num_cols:
        if col not in df.columns or col == "TransactionDT":
            continue
        if pd.api.types.is_float_dtype(df[col]):
            arr_f64 = df[col].to_numpy(dtype="float64")
            arr_f32 = arr_f64.astype("float32")
            assert np.isnan(arr_f64).sum() == np.isnan(arr_f32).sum(), f"Mất NaN ở cột {col}"
            assert np.isinf(arr_f32).sum() == 0, f"Tràn số (Inf) khi ép float32 ở cột {col}"
            df[col] = arr_f32
        elif pd.api.types.is_integer_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            for dtype_candidate in [np.int8, np.int16, np.int32]:
                info = np.iinfo(dtype_candidate)
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(dtype_candidate)
                    break

    # 3. Giữ an toàn cho 49 biến Categorical
    for col in cat_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 500:
                df[col] = df[col].astype("category")
        elif pd.api.types.is_float_dtype(df[col]):
            non_na = df[col].dropna().to_numpy(dtype="float64")
            if len(non_na) > 0 and np.array_equal(non_na, non_na.astype("float32").astype("float64")):
                df[col] = df[col].astype("float32")

    mem_after_mb = get_df_memory_mb(df)
    rss_after = get_process_rss_mb()

    stats = {
        "df_memory_before_mb": round(mem_before_mb, 2),
        "df_memory_after_mb": round(mem_after_mb, 2),
        "memory_saved_pct": round((1.0 - mem_after_mb / max(mem_before_mb, 1e-9)) * 100.0, 2),
        "process_rss_before_mb": rss_before["current_rss_mb"],
        "process_rss_after_mb": rss_after["current_rss_mb"],
        "peak_rss_mb": rss_after["peak_rss_mb"],
        "max_amt_rel_err_float32": max_amt_rel_err,
    }
    return df, stats


# ==============================================================================
# 5. CHIA TẬP TIMESTAMP-SAFE TEMPORAL SPLIT, GAP SENSITIVITY & STRATIFIED SPLIT
# ==============================================================================
def _stratified_split_indices(y: np.ndarray, train_ratio: float = 0.80, random_state: int = 42) -> np.ndarray:
    """
    Trả về mảng vị trí dòng thuộc tập Train theo phương pháp Stratified Split (bảo toàn tỷ lệ lớp).
    Sử dụng scikit-learn nếu có, hoặc dùng thuật toán phân tầng chính xác trên NumPy.
    """
    try:
        from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_ratio, random_state=random_state)
        strat_train_idx, _ = next(sss.split(np.zeros(len(y)), y))
        return strat_train_idx
    except ImportError:
        rng = np.random.default_rng(random_state)
        train_indices = []
        for cls_val in np.unique(y):
            cls_idx = np.flatnonzero(y == cls_val)
            rng.shuffle(cls_idx)
            n_cls_train = int(round(len(cls_idx) * train_ratio))
            train_indices.append(cls_idx[:n_cls_train])
        return np.concatenate(train_indices)


def create_timestamp_safe_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    gap_days_list: Tuple[int, ...] = (0, 1, 7, 14),
    n_inner_folds: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tạo và kiểm chứng ranh giới chia tập:
    1. Primary Protocol (Temporal Split 80/20): Mọi giao dịch có cùng TransactionDT tại mốc cắt
       đều được gom về cùng một phía (không bao giờ cắt ngang các dòng trùng giây).
    2. Bảng độ nhạy Gap/Embargo (0, 1, 7, 14 ngày trước mốc Validation) để mô phỏng độ trễ nhãn.
    3. Bảng 5 Inner Temporal Folds (phục vụ chọn biến ở Giai đoạn 3).
    4. Secondary Benchmark (Stratified Split 80/20) để đối chiếu với yêu cầu đề cương.
    """
    assert df["TransactionDT"].notna().all(), "Cột TransactionDT không được chứa giá trị khuyết"

    # 1. Xác định mốc cắt thời gian 80% (Timestamp-Safe)
    cut_dt = float(df["TransactionDT"].quantile(train_ratio))
    temporal_train_mask = df["TransactionDT"] <= cut_dt
    temporal_val_mask = df["TransactionDT"] > cut_dt

    # Kiểm chứng tuyệt đối không có bất kỳ giây TransactionDT nào xuất hiện ở cả Train và Val
    train_dts = set(df.loc[temporal_train_mask, "TransactionDT"].unique())
    val_dts = set(df.loc[temporal_val_mask, "TransactionDT"].unique())
    assert train_dts.isdisjoint(val_dts), "LỖI NGHIÊM TRỌNG: Có giao dịch trùng TransactionDT bị xé sang cả Train và Val!"

    # 2. Tạo Stratified Split 80/20 độc lập (chỉ dùng để đối chiếu ở bước đánh giá mô hình)
    strat_train_idx = _stratified_split_indices(
        df["isFraud"].to_numpy(), train_ratio=train_ratio, random_state=random_state
    )
    strat_role = np.full(len(df), "val", dtype=object)
    strat_role[strat_train_idx] = "train"

    split_df = pd.DataFrame(
        {
            "TransactionID": df["TransactionID"].to_numpy(),
            "TransactionDT": df["TransactionDT"].to_numpy(),
            "isFraud": df["isFraud"].to_numpy(dtype="int8"),
            "temporal_role": np.where(temporal_train_mask, "train", "val"),
            "stratified_role": strat_role,
        },
        index=df.index,
    )

    # 3. Bảng phân tích độ nhạy theo Gap / Embargo (0, 1, 7, 14 ngày)
    gap_rows = []
    val_min_dt = float(df.loc[temporal_val_mask, "TransactionDT"].min())
    val_max_dt = float(df.loc[temporal_val_mask, "TransactionDT"].max())
    val_count = int(temporal_val_mask.sum())
    val_fraud = int(df.loc[temporal_val_mask, "isFraud"].sum())

    for gap_d in gap_days_list:
        gap_sec = gap_d * 86400.0
        eff_cut_dt = cut_dt - gap_sec
        tr_mask_gap = df["TransactionDT"] <= eff_cut_dt
        gap_mask = (df["TransactionDT"] > eff_cut_dt) & (df["TransactionDT"] <= cut_dt)

        tr_cnt = int(tr_mask_gap.sum())
        tr_frd = int(df.loc[tr_mask_gap, "isFraud"].sum())
        gap_rows.append(
            {
                "Kich_ban_Gap": f"Gap = {gap_d} ngày",
                "Train_Max_Day": round(eff_cut_dt / 86400.0, 2),
                "Val_Start_Day": round(val_min_dt / 86400.0, 2),
                "Val_End_Day": round(val_max_dt / 86400.0, 2),
                "So_dong_Train": tr_cnt,
                "So_Fraud_Train": tr_frd,
                "Ty_le_Fraud_Train_pct": round(tr_frd / max(tr_cnt, 1) * 100.0, 3),
                "So_dong_trong_Gap": int(gap_mask.sum()),
                "So_dong_Val": val_count,
                "So_Fraud_Val": val_fraud,
                "Ty_le_Fraud_Val_pct": round(val_fraud / max(val_count, 1) * 100.0, 3),
            }
        )
    gap_summary_df = pd.DataFrame(gap_rows)

    # 4. Bảng thống kê 5 Inner Temporal Folds bên trong Temporal Train (Expanding Window)
    train_sub = df.loc[temporal_train_mask, ["TransactionDT", "isFraud"]].sort_values("TransactionDT")
    quantiles = np.linspace(0.0, 1.0, n_inner_folds + 2)
    dt_bounds = [float(train_sub["TransactionDT"].quantile(q)) for q in quantiles]

    fold_rows = []
    for k in range(1, n_inner_folds + 1):
        tr_end_dt = dt_bounds[k]
        va_end_dt = dt_bounds[k + 1]
        f_tr_mask = train_sub["TransactionDT"] <= tr_end_dt
        f_va_mask = (train_sub["TransactionDT"] > tr_end_dt) & (train_sub["TransactionDT"] <= va_end_dt)

        n_tr = int(f_tr_mask.sum())
        f_tr = int(train_sub.loc[f_tr_mask, "isFraud"].sum())
        n_va = int(f_va_mask.sum())
        f_va = int(train_sub.loc[f_va_mask, "isFraud"].sum())

        fold_rows.append(
            {
                "Inner_Fold": f"Fold {k}",
                "Train_Day_Range": f"{dt_bounds[0]/86400:.1f} -> {tr_end_dt/86400:.1f}",
                "Val_Day_Range": f"{tr_end_dt/86400:.1f} -> {va_end_dt/86400:.1f}",
                "Train_Rows": n_tr,
                "Train_Fraud": f_tr,
                "Train_Fraud_pct": round(f_tr / max(n_tr, 1) * 100.0, 3),
                "Val_Rows": n_va,
                "Val_Fraud": f_va,
                "Val_Fraud_pct": round(f_va / max(n_va, 1) * 100.0, 3),
            }
        )
    inner_folds_df = pd.DataFrame(fold_rows)

    return split_df, gap_summary_df, inner_folds_df


# ==============================================================================
# 6. HÀM CHẠY TRỌN GÓI BƯỚC 0 (PIPELINE ENTRYPOINT)
# ==============================================================================
def run_step0_pipeline(
    train_df: pd.DataFrame,
    output_dir: str = "./artifacts_step0",
) -> Dict[str, object]:
    """
    Thực thi trọn gói Bước 0 trên bảng dữ liệu Train đã ghép của Người 1:
    1. Kiểm định Schema 432 biến (383 Numerical + 49 Categorical).
    2. Tách Amt_cents (trên float64) và cờ has_identity.
    3. Hạ kiểu bộ nhớ an toàn và đo mức tiết kiệm RAM + Peak RSS.
    4. Tạo Timestamp-Safe Temporal Split, bảng độ nhạy Gap và bảng 5 Inner Folds.
    5. Lưu split_indices.parquet và schema_step0.json để bàn giao.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_cols, cat_cols = get_ieee_cis_canonical_schema()

    # Bước 0.1: Trích xuất Amt_cents (float64) và has_identity trước khi hạ kiểu
    train_df = extract_pre_downcast_features(train_df)

    # Kiểm tra đủ 432 biến gốc trong bảng đã ghép
    missing_schema_cols = [c for c in (num_cols + cat_cols) if c not in train_df.columns]
    assert len(missing_schema_cols) == 0, f"Thiếu các cột trong Schema 432 biến: {missing_schema_cols[:10]}"

    # Bước 0.2: Hạ kiểu RAM an toàn
    train_df, mem_stats = downcast_dataframe_safely(train_df, num_cols=num_cols, cat_cols=cat_cols)

    # Bước 0.3: Chia tập Timestamp-Safe Temporal Split & Stratified Split
    split_df, gap_summary_df, inner_folds_df = create_timestamp_safe_splits(train_df)

    # Bước 0.4: Xuất file bàn giao
    split_file = out_path / "split_indices.parquet"
    schema_file = out_path / "schema_step0.json"
    try:
        split_df.to_parquet(split_file, index=False)
    except Exception:
        split_file = out_path / "split_indices.csv"
        split_df.to_csv(split_file, index=False)

    schema_payload = {
        "n_rows_train_merged": int(len(train_df)),
        "n_candidate_features_original": 432,
        "n_numerical_original": len(num_cols),
        "n_categorical_original": len(cat_cols),
        "step0_new_features": ["has_identity", "Amt_cents"],
        "identity_coverage_rate_pct": round(float(train_df["has_identity"].mean() * 100.0), 2),
        "overall_fraud_rate_pct": round(float(train_df["isFraud"].mean() * 100.0), 2),
        "memory_stats": mem_stats,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
    }
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(schema_payload, f, indent=2, ensure_ascii=False)

    print("=" * 76)
    print("HOÀN TẤT BƯỚC 0: CHUẨN BỊ DỮ LIỆU, HẠ KIỂU RAM & CHIA TẬP AN TOÀN")
    print("=" * 76)
    print(f"- Kích thước bảng Train: {train_df.shape[0]:,} dòng x {train_df.shape[1]} cột (đã thêm has_identity, Amt_cents)")
    print(f"- Định nghĩa Schema gốc: {len(num_cols)} biến Numerical + {len(cat_cols)} biến Categorical = 432 biến")
    print(
        f"- Dung lượng DataFrame : {mem_stats['df_memory_before_mb']:.2f} MB -> "
        f"{mem_stats['df_memory_after_mb']:.2f} MB (Tiết kiệm {mem_stats['memory_saved_pct']:.2f}%)"
    )
    print(
        f"- Bộ nhớ Tiến trình    : RSS hiện tại = {mem_stats['process_rss_after_mb']:.2f} MB | "
        f"Peak RSS = {mem_stats['peak_rss_mb']:.2f} MB"
    )
    print(
        f"- Sai số tương đối tối đa của TransactionAmt khi sang float32: "
        f"{mem_stats['max_amt_rel_err_float32']:.2e}"
    )
    print(f"- Tỷ lệ khớp Identity  : {schema_payload['identity_coverage_rate_pct']:.2f}%")
    print(f"- Tỷ lệ Fraud toàn tập : {schema_payload['overall_fraud_rate_pct']:.2f}%")
    print("\n[Bảng 1] Phân tích độ nhạy theo kịch bản độ trễ nhãn (Gap / Embargo):")
    print(gap_summary_df.to_string(index=False))
    print("\n[Bảng 2] Thống kê 5 Inner Temporal Folds (phục vụ chọn biến ở Giai đoạn 3):")
    print(inner_folds_df.to_string(index=False))
    print(f"\nĐã lưu file chỉ số chia tập tại: {split_file}")
    print(f"Đã lưu file đặc tả Schema tại  : {schema_file}")

    return {
        "df": train_df,
        "split_df": split_df,
        "gap_summary_df": gap_summary_df,
        "inner_folds_df": inner_folds_df,
        "schema_payload": schema_payload,
    }


# ==============================================================================
# 7. BỘ KIỂM THỬ TỰ ĐỘNG CHO BƯỚC 0 (SELF-TEST CHẠY ĐỘC LẬP)
# ==============================================================================
def _build_synthetic_ieee_cis_sample(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Tạo dữ liệu giả lập đúng cấu trúc 434 cột của IEEE-CIS để kiểm thử tự động Bước 0."""
    rng = np.random.default_rng(seed)
    num_cols, cat_cols = get_ieee_cis_canonical_schema()

    data: Dict[str, object] = {
        "TransactionID": np.arange(2987000, 2987000 + n_rows, dtype=np.int64),
        "isFraud": rng.choice([0, 1], size=n_rows, p=[0.965, 0.035]).astype(np.int64),
    }

    # Tạo TransactionDT tăng dần có các dòng trùng giây (timestamp ties) ở ngay mốc phân vị 80%
    base_dt = np.sort(rng.integers(86400, 86400 * 180, size=n_rows)).astype(np.int64)
    cut_pos = int(n_rows * 0.80)
    base_dt[cut_pos - 3 : cut_pos + 3] = base_dt[cut_pos]  # Cố tình tạo 6 giao dịch trùng giây ở ranh giới cắt
    data["TransactionDT"] = base_dt

    # Tạo TransactionAmt có phần xu lẻ chính xác (ví dụ 49.95, 1024.37)
    dollars = rng.integers(1, 5000, size=n_rows).astype(np.float64)
    cents = rng.integers(0, 100, size=n_rows).astype(np.float64) / 100.0
    data["TransactionAmt"] = dollars + cents

    for col in num_cols:
        if col in ("TransactionDT", "TransactionAmt"):
            continue
        vals = rng.normal(loc=10.0, scale=5.0, size=n_rows).astype(np.float64)
        vals[rng.random(n_rows) < 0.2] = np.nan
        data[col] = vals

    has_id_mask = rng.random(n_rows) < 0.2442

    for col in cat_cols:
        if col.startswith("id_") or col in ("DeviceType", "DeviceInfo"):
            vals_cat = np.where(has_id_mask, rng.choice(["A", "B", "C"], size=n_rows), None)
            data[col] = pd.Series(vals_cat, dtype="object")
        elif col in ("card1", "card2", "card3", "card5", "addr1", "addr2"):
            vals_num_cat = rng.integers(100, 600, size=n_rows).astype(np.float64)
            vals_num_cat[rng.random(n_rows) < 0.05] = np.nan
            data[col] = vals_num_cat
        else:
            data[col] = pd.Series(rng.choice(["W", "H", "C", None], size=n_rows), dtype="object")

    for i in range(1, 12):
        col = f"id_{i:02d}"
        arr = data[col].copy()
        arr[~has_id_mask] = np.nan
        data[col] = arr

    return pd.DataFrame(data)


if __name__ == "__main__":
    sample_df = _build_synthetic_ieee_cis_sample(n_rows=2000)
    expected_cents = np.round(sample_df["TransactionAmt"].to_numpy(dtype="float64") * 100).astype("int64") % 100

    result = run_step0_pipeline(sample_df, output_dir="/tmp/test_step0_artifacts")

    assert np.array_equal(result["df"]["Amt_cents"].to_numpy(), expected_cents), "Sai lệch Amt_cents!"
    sp = result["split_df"]
    tr_t = set(sp.loc[sp["temporal_role"] == "train", "TransactionDT"])
    va_t = set(sp.loc[sp["temporal_role"] == "val", "TransactionDT"])
    assert tr_t.isdisjoint(va_t), "Ranh giới Temporal Split làm xé đôi giao dịch trùng giây!"
    print("\n[SELF-TEST PASSED] Tất cả kiểm thử tự động của Bước 0 đều vượt qua 100%!")