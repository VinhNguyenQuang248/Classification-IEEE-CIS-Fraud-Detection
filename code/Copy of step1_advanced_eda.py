"""
Bước 1 (Giai đoạn 1): Phân tích Khám phá Chuyên sâu (Advanced EDA) trên tập Temporal Train.

Đặc điểm kỹ thuật:
1. Chống Data Leakage tuyệt đối: Mọi thống kê và biểu đồ có sử dụng nhãn isFraud chỉ được tính
   trên tập Temporal Train (80% thời gian đầu từ split_indices.parquet của Bước 0).
2. Phân tích Khuyết có điều kiện (Conditional Missingness):
   - Tách riêng đánh giá 40 biến Identity bên trong tập con has_identity == 1.
   - Gom cụm động 339 biến V1-V339 theo mặt nạ khuyết thực tế bằng mã băm 64-bit (không tốn RAM).
   - Đánh giá chênh lệch gian lận bằng Risk Ratio (Lift) so với nhóm bù, khoảng tin cậy 95% Wilson
     và hiệu chỉnh đa kiểm định Benjamini-Hochberg (FDR).
3. Phân tích Biến Phân loại & Biến Số theo lớp (isFraud = 0 vs 1):
   - Mọi bảng/biểu đồ Fraud Rate đều đi kèm Số lượng giao dịch (Count), Số ca Fraud và CI 95%.
   - Tính Cramér's V có hiệu chỉnh thiên lệch (Bias-corrected Cramér's V) cho 49 biến phân loại.
4. Phân tích Tương quan Hạng (Pairwise-Complete Spearman trước khi điền khuyết) & Ngoại lai (95th, 99th).
"""

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_setup_and_split import (
    _build_synthetic_ieee_cis_sample,
    get_ieee_cis_canonical_schema,
    run_step0_pipeline,
)


# ==============================================================================
# 1. CÁC HÀM THỐNG KÊ CỐT LÕI (WILSON 95% CI, LIFT NHÓM BÙ, CRAMÉR'S V & FDR)
# ==============================================================================
def wilson_ci_95(k: np.ndarray, n: np.ndarray, z: float = 1.959963984540054) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính khoảng tin cậy 95% theo công thức Wilson cho tỷ lệ nhị thức p = k / n.
    Chính xác hơn xấp xỉ chuẩn truyền thống khi tỷ lệ Fraud nhỏ (3.5%) hoặc cỡ mẫu nhóm nhỏ.
    """
    k_arr = np.asarray(k, dtype="float64")
    n_arr = np.asarray(n, dtype="float64")
    safe_n = np.maximum(n_arr, 1.0)
    p = k_arr / safe_n
    z2 = z * z

    denom = 1.0 + z2 / safe_n
    center = (p + z2 / (2.0 * safe_n)) / denom
    half_width = (z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * safe_n)) / safe_n)) / denom

    low = np.where(n_arr > 0, np.clip(center - half_width, 0.0, 1.0), np.nan)
    high = np.where(n_arr > 0, np.clip(center + half_width, 0.0, 1.0), np.nan)
    return low, high


def benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """Hiệu chỉnh đa kiểm định theo thủ tục Benjamini-Hochberg (kiểm soát False Discovery Rate)."""
    p = np.asarray(p_values, dtype="float64")
    q = np.full_like(p, np.nan, dtype="float64")
    valid_mask = np.isfinite(p)
    p_valid = p[valid_mask]
    m = len(p_valid)
    if m == 0:
        return q

    order = np.argsort(p_valid)
    ranked_p = p_valid[order]
    ranks = np.arange(1, m + 1, dtype="float64")
    q_ranked = ranked_p * (m / ranks)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q_valid = np.empty_like(p_valid)
    q_valid[order] = q_ranked
    q[valid_mask] = q_valid
    return q


def two_proportion_z_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    """Tính p-value kiểm định 2 tỷ lệ độc lập (nhóm đang xét vs nhóm bù)."""
    if n1 <= 0 or n2 <= 0:
        return np.nan
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool <= 0.0 or p_pool >= 1.0:
        return 1.0
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se <= 0.0:
        return 1.0
    z = abs(p1 - p2) / se
    return float(erfc(z / sqrt(2.0)))


def bias_corrected_cramers_v(cat_series: pd.Series, y: pd.Series) -> float:
    """
    Tính hệ số Cramér's V có hiệu chỉnh thiên lệch (Bergsma 2013) giữa biến phân loại và isFraud.
    Bao gồm cả trạng thái 'Missing' như một nhóm riêng biệt để không bỏ sót tín hiệu khuyết.
    """
    s = cat_series.astype("object").fillna("__MISSING__").astype(str)
    ct = pd.crosstab(s, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0

    obs = ct.to_numpy(dtype="float64")
    n = obs.sum()
    if n <= 1:
        return 0.0

    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / n
    valid = expected > 0
    chi2 = float(np.sum(((obs[valid] - expected[valid]) ** 2) / expected[valid]))

    phi2 = chi2 / n
    r, k = obs.shape
    phi2_corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / (n - 1.0))
    r_corr = r - ((r - 1.0) ** 2) / (n - 1.0)
    k_corr = k - ((k - 1.0) ** 2) / (n - 1.0)
    denom = min(k_corr - 1.0, r_corr - 1.0)
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


# ==============================================================================
# 2. PHÂN TÍCH ĐỘ PHỦ IDENTITY, KHUYẾT CÓ ĐIỀU KIỆN & GOM CỤM HỌ V
# ==============================================================================
def analyze_conditional_missingness(
    train_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Thực hiện 3 phân tích cấu trúc khuyết trên tập Temporal Train:
    1. Bảng so sánh tỷ lệ gian lận theo sự hiện diện của bảng Identity (has_identity = 1 vs 0).
    2. Bảng đánh giá tỷ lệ khuyết và Risk Ratio (Lift) của từng biến:
       - Các biến Transaction (392 biến): đánh giá trên toàn bộ Temporal Train.
       - Các biến Identity (40 biến): đánh giá CÓ ĐIỀU KIỆN bên trong nhóm has_identity == 1.
       - Áp dụng hiệu chỉnh đa kiểm định Benjamini-Hochberg (FDR).
    3. Gom cụm động các biến V1-V339 có cùng mặt nạ khuyết thực tế (missingness pattern).
    """
    y_all = train_df["isFraud"].to_numpy(dtype="int64")
    n_total = len(train_df)
    total_fraud = int(y_all.sum())

    # 1. Phân tích Identity Coverage (has_identity == 1 vs 0)
    id_cov_rows = []
    for val, label in [(1, "Có bản ghi Identity (has_identity = 1)"), (0, "Không có Identity (has_identity = 0)")]:
        mask = (train_df["has_identity"].to_numpy() == val)
        n_grp = int(mask.sum())
        k_grp = int(y_all[mask].sum())
        n_comp = n_total - n_grp
        k_comp = total_fraud - k_grp

        rate_grp = k_grp / max(n_grp, 1)
        rate_comp = k_comp / max(n_comp, 1)
        ci_low, ci_high = wilson_ci_95(np.array([k_grp]), np.array([n_grp]))
        lift = rate_grp / max(rate_comp, 1e-9)

        id_cov_rows.append(
            {
                "Nhom_Identity": label,
                "So_luong_Giao_dich": n_grp,
                "Ty_trong_pct": round(n_grp / max(n_total, 1) * 100.0, 2),
                "So_ca_Fraud": k_grp,
                "Fraud_Rate_pct": round(rate_grp * 100.0, 3),
                "CI_95_Low_pct": round(float(ci_low[0]) * 100.0, 3),
                "CI_95_High_pct": round(float(ci_high[0]) * 100.0, 3),
                "Chenh_lech_vs_Nhom_bu_pp": round((rate_grp - rate_comp) * 100.0, 3),
                "Risk_Ratio_Lift": round(lift, 3),
            }
        )
    identity_coverage_df = pd.DataFrame(id_cov_rows)

    # 2. Đánh giá Missingness & Lift có điều kiện cho 432 biến gốc
    identity_set = set([f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"])
    id_sub_mask = (train_df["has_identity"].to_numpy() == 1)

    miss_rows = []
    for col in (num_cols + cat_cols):
        if col not in train_df.columns:
            continue
        is_id_col = col in identity_set
        if is_id_col:
            sub_y = y_all[id_sub_mask]
            is_miss = train_df.loc[id_sub_mask, col].isna().to_numpy()
            scope_name = "Conditional (has_identity=1)"
        else:
            sub_y = y_all
            is_miss = train_df[col].isna().to_numpy()
            scope_name = "Full Temporal Train"

        n_scope = len(sub_y)
        n_miss = int(is_miss.sum())
        n_pres = n_scope - n_miss
        k_miss = int(sub_y[is_miss].sum())
        k_pres = int(sub_y[~is_miss].sum())

        r_miss = k_miss / max(n_miss, 1) if n_miss > 0 else np.nan
        r_pres = k_pres / max(n_pres, 1) if n_pres > 0 else np.nan
        lift_miss_vs_pres = (r_miss / max(r_pres, 1e-9)) if (n_miss > 0 and n_pres > 0 and r_pres > 0) else np.nan
        diff_pp = (r_miss - r_pres) * 100.0 if (n_miss > 0 and n_pres > 0) else np.nan
        pval = two_proportion_z_pvalue(k_miss, n_miss, k_pres, n_pres) if (n_miss >= 10 and n_pres >= 10) else np.nan

        miss_rows.append(
            {
                "Feature": col,
                "Scope": scope_name,
                "Missing_Count": n_miss,
                "Present_Count": n_pres,
                "Missing_Rate_pct": round(n_miss / max(n_scope, 1) * 100.0, 2),
                "Fraud_Rate_Missing_pct": round(r_miss * 100.0, 3) if np.isfinite(r_miss) else np.nan,
                "Fraud_Rate_Present_pct": round(r_pres * 100.0, 3) if np.isfinite(r_pres) else np.nan,
                "Diff_Missing_vs_Present_pp": round(diff_pp, 3) if np.isfinite(diff_pp) else np.nan,
                "Lift_Missing_vs_Present": round(lift_miss_vs_pres, 3) if np.isfinite(lift_miss_vs_pres) else np.nan,
                "p_value_raw": pval,
            }
        )

    missing_audit_df = pd.DataFrame(miss_rows)
    missing_audit_df["q_value_bh_fdr"] = benjamini_hochberg_fdr(missing_audit_df["p_value_raw"].to_numpy())
    missing_audit_df["Has_Strong_Missing_Signal"] = (
        (missing_audit_df["q_value_bh_fdr"] < 0.05)
        & (
            (missing_audit_df["Lift_Missing_vs_Present"] >= 1.25)
            | (missing_audit_df["Lift_Missing_vs_Present"] <= 0.80)
            | (missing_audit_df["Diff_Missing_vs_Present_pp"].abs() >= 1.0)
        )
    )
    missing_audit_df = missing_audit_df.sort_values("Missing_Rate_pct", ascending=False).reset_index(drop=True)

    # 3. Gom cụm động các biến V1-V339 có cùng mặt nạ khuyết thực tế (Missingness Pattern Clustering)
    v_cols = [f"V{i}" for i in range(1, 340) if f"V{i}" in train_df.columns]
    pattern_groups: Dict[Tuple[int, int], List[str]] = {}
    for col in v_cols:
        mask_arr = train_df[col].isna().to_numpy(dtype=np.uint8)
        n_na = int(mask_arr.sum())
        mask_hash = int(pd.util.hash_array(mask_arr).sum())
        key = (n_na, mask_hash)
        pattern_groups.setdefault(key, []).append(col)

    v_cluster_rows = []
    for cluster_idx, ((n_na, _), cols_in_cluster) in enumerate(
        sorted(pattern_groups.items(), key=lambda x: x[0][0]), start=1
    ):
        v_cluster_rows.append(
            {
                "Cluster_ID": f"V_Cluster_{cluster_idx:02d}",
                "Num_V_Columns": len(cols_in_cluster),
                "Missing_Count": n_na,
                "Missing_Rate_pct": round(n_na / max(n_total, 1) * 100.0, 2),
                "Sample_Columns": ", ".join(cols_in_cluster[:6]) + ("..." if len(cols_in_cluster) > 6 else ""),
                "All_Columns": cols_in_cluster,
            }
        )
    v_clusters_df = pd.DataFrame(v_cluster_rows)

    return identity_coverage_df, missing_audit_df, v_clusters_df


# ==============================================================================
# 3. PHÂN TÍCH BIẾN SỐ THEO NHÃN ISFRAUD, QUAN HỆ D-TIME & NGOẠI LAI (95TH/99TH)
# ==============================================================================
def analyze_numerical_and_outliers(
    train_df: pd.DataFrame,
    focus_num_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Thực hiện 3 bảng phân tích cho biến số trên tập Temporal Train:
    1. Thống kê phân phối (Mean, Std, Median, IQR, 95th, 99th) tách theo Non-Fraud (0) và Fraud (1).
    2. Phân tích tỷ lệ gian lận ở vùng ngoại lai (trên phân vị thứ 95 và 99) so với nhóm bù.
    3. Kiểm tra mối tương quan giữa D1-D15 và thời gian trôi qua (elapsed_day) để làm cơ sở cho D_anchor.
    """
    if focus_num_cols is None:
        focus_num_cols = ["TransactionAmt", "Amt_cents", "dist1", "C1", "C2", "C13", "D1", "D2", "D15"]
    focus_num_cols = [c for c in focus_num_cols if c in train_df.columns]

    y = train_df["isFraud"].to_numpy(dtype="int64")
    total_n = len(train_df)

    dist_rows = []
    outlier_rows = []

    for col in focus_num_cols:
        s_all = train_df[col].dropna()
        if len(s_all) == 0:
            continue
        for cls_val, cls_name in [(0, "Non-Fraud (0)"), (1, "Fraud (1)")]:
            s_cls = train_df.loc[train_df["isFraud"] == cls_val, col].dropna()
            if len(s_cls) == 0:
                continue
            q25, q50, q75, q95, q99 = np.percentile(s_cls.to_numpy(dtype="float64"), [25, 50, 75, 95, 99])
            dist_rows.append(
                {
                    "Feature": col,
                    "Class": cls_name,
                    "Observed_Count": len(s_cls),
                    "Mean": round(float(s_cls.mean()), 3),
                    "Std": round(float(s_cls.std()), 3),
                    "Median": round(float(q50), 3),
                    "IQR": round(float(q75 - q25), 3),
                    "P95": round(float(q95), 3),
                    "P99": round(float(q99), 3),
                    "Max": round(float(s_cls.max()), 3),
                }
            )

        # Phân tích Outliers ở ngưỡng P95 và P99 trên các dòng có giá trị quan sát
        obs_mask = train_df[col].notna().to_numpy()
        vals_obs = train_df.loc[obs_mask, col].to_numpy(dtype="float64")
        y_obs = y[obs_mask]
        n_obs = len(y_obs)
        k_obs = int(y_obs.sum())

        p95_thr, p99_thr = np.percentile(vals_obs, [95, 99])
        for pct_label, thr in [("P95 (Top 5%)", p95_thr), ("P99 (Top 1%)", p99_thr)]:
            out_mask = vals_obs >= thr
            n_out = int(out_mask.sum())
            k_out = int(y_obs[out_mask].sum())
            n_comp = n_obs - n_out
            k_comp = k_obs - k_out

            r_out = k_out / max(n_out, 1)
            r_comp = k_comp / max(n_comp, 1)
            ci_low, ci_high = wilson_ci_95(np.array([k_out]), np.array([n_out]))
            lift = r_out / max(r_comp, 1e-9)

            outlier_rows.append(
                {
                    "Feature": col,
                    "Percentile_Gate": pct_label,
                    "Threshold_Value": round(float(thr), 3),
                    "Outlier_Count": n_out,
                    "Outlier_Fraud_Count": k_out,
                    "Outlier_Fraud_Rate_pct": round(r_out * 100.0, 3),
                    "CI_95_Low_pct": round(float(ci_low[0]) * 100.0, 3),
                    "CI_95_High_pct": round(float(ci_high[0]) * 100.0, 3),
                    "Complement_Fraud_Rate_pct": round(r_comp * 100.0, 3),
                    "Lift_vs_Complement": round(lift, 3),
                }
            )

    num_dist_df = pd.DataFrame(dist_rows)
    outlier_df = pd.DataFrame(outlier_rows)

    # 3. Phân tích quan hệ giữa D1-D15 và thời gian trôi qua (elapsed_day)
    elapsed_day = (train_df["TransactionDT"].to_numpy(dtype="float64") // 86400.0)
    d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in train_df.columns]
    d_rows = []
    for d_col in d_cols:
        d_vals = train_df[d_col].to_numpy(dtype="float64")
        valid = np.isfinite(d_vals)
        n_val = int(valid.sum())
        if n_val < 10:
            continue
        anchor_vals = elapsed_day[valid] - d_vals[valid]
        corr_dt = float(pd.Series(d_vals[valid]).corr(pd.Series(elapsed_day[valid]), method="spearman"))
        neg_rate = float((d_vals[valid] < 0).mean() * 100.0)

        d_rows.append(
            {
                "D_Feature": d_col,
                "Observed_Count": n_val,
                "Missing_Rate_pct": round((1.0 - n_val / max(total_n, 1)) * 100.0, 2),
                "Negative_Values_pct": round(neg_rate, 3),
                "Spearman_with_ElapsedDay": round(corr_dt, 4),
                "Raw_D_Std": round(float(np.std(d_vals[valid])), 2),
                "Anchor_D_Std": round(float(np.std(anchor_vals)), 2),
            }
        )
    d_drift_df = pd.DataFrame(d_rows)

    return num_dist_df, outlier_df, d_drift_df


# ==============================================================================
# 4. PHÂN TÍCH BIẾN PHÂN LOẠI (COUNT + FRAUD RATE + CI 95% + CRAMÉR'S V)
# ==============================================================================
def analyze_categorical_features(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    key_cat_cols: Optional[List[str]] = None,
    top_k_levels: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Tính hệ số Cramér's V có hiệu chỉnh thiên lệch cho toàn bộ 49 biến phân loại.
    2. Lập bảng chi tiết (Count, Fraud Count, Fraud Rate %, Wilson CI 95%, Lift vs nhóm bù)
       cho các biến phân loại trọng tâm (ProductCD, card4, card6, P_emaildomain, DeviceType, M1-M9).
    """
    y_series = train_df["isFraud"]
    y_arr = y_series.to_numpy(dtype="int64")
    total_n = len(train_df)
    total_k = int(y_arr.sum())

    # 1. Xếp hạng Cramér's V cho toàn bộ 49 biến phân loại
    cramers_rows = []
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        v_score = bias_corrected_cramers_v(train_df[col], y_series)
        n_uniq = int(train_df[col].nunique(dropna=True))
        miss_pct = float(train_df[col].isna().mean() * 100.0)
        cramers_rows.append(
            {
                "Categorical_Feature": col,
                "Unique_Levels": n_uniq,
                "Missing_Rate_pct": round(miss_pct, 2),
                "Cramers_V_BiasCorrected": round(v_score, 4),
            }
        )
    cramers_df = pd.DataFrame(cramers_rows).sort_values("Cramers_V_BiasCorrected", ascending=False).reset_index(drop=True)

    # 2. Bảng chi tiết theo từng giá trị (Level) của các biến phân loại chủ chốt
    if key_cat_cols is None:
        key_cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "M1", "M4"]
    key_cat_cols = [c for c in key_cat_cols if c in train_df.columns]

    level_rows = []
    for col in key_cat_cols:
        s = train_df[col].astype("object").fillna("Missing").astype(str)
        top_levels = s.value_counts().head(top_k_levels).index.tolist()
        s_grouped = np.where(s.isin(top_levels), s, "Other_Rare")

        for lvl in pd.Series(s_grouped).value_counts().index:
            mask = (s_grouped == lvl)
            n_lvl = int(mask.sum())
            k_lvl = int(y_arr[mask].sum())
            n_comp = total_n - n_lvl
            k_comp = total_k - k_lvl

            r_lvl = k_lvl / max(n_lvl, 1)
            r_comp = k_comp / max(n_comp, 1) if n_comp > 0 else total_k / max(total_n, 1)
            ci_low, ci_high = wilson_ci_95(np.array([k_lvl]), np.array([n_lvl]))
            lift = r_lvl / max(r_comp, 1e-9)

            level_rows.append(
                {
                    "Feature": col,
                    "Category_Level": str(lvl),
                    "Transaction_Count": n_lvl,
                    "Share_pct": round(n_lvl / max(total_n, 1) * 100.0, 2),
                    "Fraud_Count": k_lvl,
                    "Fraud_Rate_pct": round(r_lvl * 100.0, 3),
                    "CI_95_Low_pct": round(float(ci_low[0]) * 100.0, 3),
                    "CI_95_High_pct": round(float(ci_high[0]) * 100.0, 3),
                    "Lift_vs_Complement": round(lift, 3),
                }
            )
    cat_levels_df = pd.DataFrame(level_rows)
    return cramers_df, cat_levels_df


# ==============================================================================
# 5. TƯƠNG QUAN HẠNG SPEARMAN TRƯỚC KHI ĐIỀN KHUYẾT (PAIRWISE-COMPLETE)
# ==============================================================================
def analyze_spearman_correlations(
    train_df: pd.DataFrame,
    num_cols: List[str],
    min_observed_pairs: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Tính tương quan hạng Spearman giữa từng biến Numerical (chưa điền khuyết) và isFraud.
    2. Phát hiện các cặp biến có đa cộng tuyến cao (|rho| >= 0.90) bên trong họ C và D.
    """
    y_s = train_df["isFraud"].astype("float64")
    target_corr_rows = []

    for col in num_cols:
        if col not in train_df.columns:
            continue
        s = train_df[col]
        valid = s.notna()
        n_obs = int(valid.sum())
        if n_obs < min_observed_pairs or s[valid].nunique() < 2:
            continue
        rho = float(s[valid].astype("float64").corr(y_s[valid], method="spearman"))
        target_corr_rows.append(
            {
                "Feature": col,
                "Observed_Count": n_obs,
                "Spearman_with_isFraud": round(rho, 4),
                "Abs_Spearman": round(abs(rho), 4),
            }
        )

    target_corr_df = (
        pd.DataFrame(target_corr_rows, columns=["Feature", "Observed_Count", "Spearman_with_isFraud", "Abs_Spearman"])
        .sort_values("Abs_Spearman", ascending=False)
        .reset_index(drop=True)
    )

    # Kiểm tra các cặp tương quan cao (|rho| >= 0.90) trong nhóm C1-C14 và D1-D15
    cd_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in train_df.columns] + [
        f"D{i}" for i in range(1, 16) if f"D{i}" in train_df.columns
    ]
    corr_mat_cd = train_df[cd_cols].astype("float64").corr(method="spearman")
    high_pairs = []
    for i in range(len(cd_cols)):
        for j in range(i + 1, len(cd_cols)):
            c1, c2 = cd_cols[i], cd_cols[j]
            val = corr_mat_cd.loc[c1, c2]
            if np.isfinite(val) and abs(val) >= 0.90:
                high_pairs.append(
                    {
                        "Feature_1": c1,
                        "Feature_2": c2,
                        "Spearman_Corr": round(float(val), 4),
                    }
                )
    high_collinear_df = (
        pd.DataFrame(high_pairs, columns=["Feature_1", "Feature_2", "Spearman_Corr"])
        .sort_values("Spearman_Corr", ascending=False)
        .reset_index(drop=True)
    )

    return target_corr_df, high_collinear_df


# ==============================================================================
# 6. TRỰC QUAN HÓA CHUYÊN SÂU (XUẤT 3 HÌNH CHUẨN BÁO CÁO)
# ==============================================================================
def generate_step1_figures(
    train_df: pd.DataFrame,
    identity_coverage_df: pd.DataFrame,
    cat_levels_df: pd.DataFrame,
    target_corr_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, str]:
    """Vẽ và lưu 3 biểu đồ tổng hợp cho Báo cáo Phần 2 (Advanced EDA)."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    fig_paths: Dict[str, str] = {}

    # --------------------------------------------------------------------------
    # FIGURE 4: Phân phối Numerical theo lớp isFraud (Amt_log, Elapsed Hour, C1, D1)
    # --------------------------------------------------------------------------
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    non_fraud = train_df[train_df["isFraud"] == 0]
    fraud = train_df[train_df["isFraud"] == 1]

    # Panel A: log1p(TransactionAmt)
    ax = axes[0, 0]
    ax.hist(
        np.log1p(non_fraud["TransactionAmt"].dropna()),
        bins=45,
        density=True,
        alpha=0.55,
        color="#1f77b4",
        label=f"Non-Fraud (n={len(non_fraud):,})",
    )
    ax.hist(
        np.log1p(fraud["TransactionAmt"].dropna()),
        bins=45,
        density=True,
        alpha=0.60,
        color="#d62728",
        label=f"Fraud (n={len(fraud):,})",
    )
    ax.set_title("A. Phân phối log(1 + TransactionAmt) theo lớp isFraud", fontsize=11, fontweight="bold")
    ax.set_xlabel("log(1 + TransactionAmt)")
    ax.set_ylabel("Mật độ (Density)")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Tỷ lệ gian lận và Số lượng giao dịch theo Giờ tương đối (elapsed_hour 0..23)
    ax = axes[0, 1]
    elapsed_hour = (train_df["TransactionDT"].to_numpy(dtype="int64") // 3600) % 24
    hr_df = (
        pd.DataFrame({"hour": elapsed_hour, "isFraud": train_df["isFraud"].to_numpy()})
        .groupby("hour")["isFraud"]
        .agg(cnt="count", frd="sum")
        .reindex(range(24), fill_value=0)
    )
    hr_rate = (hr_df["frd"] / np.maximum(hr_df["cnt"], 1)) * 100.0
    ci_l, ci_h = wilson_ci_95(hr_df["frd"].to_numpy(), hr_df["cnt"].to_numpy())

    ax.bar(hr_df.index, hr_df["cnt"], color="#aec7e8", alpha=0.7, label="Số giao dịch (Count)")
    ax.set_xlabel("Chu kỳ giờ tương đối (elapsed_hour: 0..23)")
    ax.set_ylabel("Số lượng giao dịch (Count)", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(hr_df.index, hr_rate, color="#d62728", marker="o", linewidth=2, label="Fraud Rate (%)")
    ax2.fill_between(hr_df.index, ci_l * 100.0, ci_h * 100.0, color="#d62728", alpha=0.18, label="95% Wilson CI")
    ax2.set_ylabel("Tỷ lệ gian lận - Fraud Rate (%)", color="#d62728")
    ax.set_title("B. Số lượng giao dịch & Fraud Rate theo Chu kỳ 24 giờ", fontsize=11, fontweight="bold")

    # Panel C: Boxplot log(1 + C1) theo lớp
    ax = axes[1, 0]
    c1_0 = np.log1p(non_fraud["C1"].dropna().clip(lower=0))
    c1_1 = np.log1p(fraud["C1"].dropna().clip(lower=0))
    bp = ax.boxplot(
        [c1_0, c1_1],
        tick_labels=["Non-Fraud (0)", "Fraud (1)"],
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], ["#9ecae1", "#fcbba1"]):
        patch.set_facecolor(color)
    ax.set_title("C. So sánh phân phối log(1 + C1) giữa 2 nhóm", fontsize=11, fontweight="bold")
    ax.set_ylabel("log(1 + C1)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel D: Phân phối phần xu lẻ (Amt_cents = 0 vs > 0)
    ax = axes[1, 1]
    is_nonzero_cents = np.where(train_df["Amt_cents"] == 0, "Số tiền chẵn (.00)", "Có xu lẻ (.01-.99)")
    cents_grp = (
        pd.DataFrame({"grp": is_nonzero_cents, "isFraud": train_df["isFraud"].to_numpy()})
        .groupby("grp")["isFraud"]
        .agg(cnt="count", frd="sum")
    )
    cents_rate = (cents_grp["frd"] / np.maximum(cents_grp["cnt"], 1)) * 100.0
    bars = ax.bar(cents_grp.index, cents_rate, color=["#2ca02c", "#ff7f0e"], width=0.45)
    for bar, (_, row_val) in zip(bars, cents_grp.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.1,
            f"{h:.2f}%\n(n={int(row_val['cnt']):,})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0, max(float(cents_rate.max()) * 1.35, 5.0))
    ax.set_title("D. Tỷ lệ gian lận theo Phần xu lẻ (Amt_cents)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fraud Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig4.tight_layout()
    fig4_path = output_dir / "fig4_numerical_distributions_by_fraud.png"
    fig4.savefig(fig4_path)
    plt.close(fig4)
    fig_paths["fig4_numerical"] = str(fig4_path)

    # --------------------------------------------------------------------------
    # FIGURE 5: Tỷ lệ gian lận kèm Số lượng giao dịch cho Biến Phân loại & Identity
    # --------------------------------------------------------------------------
    fig5, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Panel A: Identity Coverage
    ax = axes[0]
    labels_id = ["Có Identity\n(has_identity=1)", "Không có Identity\n(has_identity=0)"]
    rates_id = identity_coverage_df["Fraud_Rate_pct"].to_numpy()
    cnts_id = identity_coverage_df["So_luong_Giao_dich"].to_numpy()
    bars = ax.bar(labels_id, rates_id, color=["#d62728", "#1f77b4"], width=0.45)
    for bar, r_val, c_val in zip(bars, rates_id, cnts_id):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            r_val + 0.15,
            f"{r_val:.2f}%\n(n={int(c_val):,})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_ylim(0, max(float(rates_id.max()) * 1.35, 6.0))
    ax.set_title("A. Chênh lệch Fraud Rate theo Độ phủ bảng Identity", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fraud Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: ProductCD (Count + Fraud Rate %)
    ax = axes[1]
    prod_df = cat_levels_df[cat_levels_df["Feature"] == "ProductCD"].copy()
    if len(prod_df) > 0:
        x_pos = np.arange(len(prod_df))
        ax.bar(x_pos, prod_df["Transaction_Count"], color="#c6dbef", width=0.5, label="Transaction Count")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(prod_df["Category_Level"])
        ax.set_ylabel("Số lượng giao dịch (Count)", color="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot(
            x_pos,
            prod_df["Fraud_Rate_pct"],
            color="#d62728",
            marker="s",
            linewidth=2.2,
            label="Fraud Rate (%)",
        )
        for x_i, r_i, f_i in zip(x_pos, prod_df["Fraud_Rate_pct"], prod_df["Fraud_Count"]):
            ax2.text(x_i, r_i + 0.2, f"{r_i:.1f}%\n({int(f_i)} ca)", ha="center", va="bottom", fontsize=8.5, color="#b30000")
        ax2.set_ylim(0, max(float(prod_df["Fraud_Rate_pct"].max()) * 1.35, 6.0))
        ax2.set_ylabel("Fraud Rate (%)", color="#d62728")
        ax.set_title("B. Số lượng giao dịch & Fraud Rate theo Mã sản phẩm (ProductCD)", fontsize=11, fontweight="bold")

    fig5.tight_layout()
    fig5_path = output_dir / "fig5_categorical_fraud_rate_with_counts.png"
    fig5.savefig(fig5_path)
    plt.close(fig5)
    fig_paths["fig5_categorical"] = str(fig5_path)

    # --------------------------------------------------------------------------
    # FIGURE 6: Top Tương quan Spearman với isFraud & Lift ở ngưỡng Ngoại lai P95/P99
    # --------------------------------------------------------------------------
    fig6, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    ax = axes[0]
    top_corr = target_corr_df.head(12).iloc[::-1]
    colors_corr = np.where(top_corr["Spearman_with_isFraud"] >= 0, "#d62728", "#1f77b4")
    ax.barh(top_corr["Feature"], top_corr["Spearman_with_isFraud"], color=colors_corr)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title("A. Top 12 biến Numerical có tương quan Spearman mạnh nhất với isFraud", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Hệ số tương quan hạng Spearman (trước khi điền khuyết)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax = axes[1]
    p99_sub = outlier_df[outlier_df["Percentile_Gate"] == "P99 (Top 1%)"].head(8).iloc[::-1]
    if len(p99_sub) > 0:
        ax.barh(p99_sub["Feature"], p99_sub["Lift_vs_Complement"], color="#ff7f0e")
        ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.2, label="Mức nền (Lift = 1.0)")
        ax.set_title("B. Hệ số nhân rủi ro (Fraud Lift) ở nhóm Ngoại lai P99 (Top 1%)", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Risk Ratio / Lift so với 99% giao dịch còn lại")
        ax.legend(frameon=True)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig6.tight_layout()
    fig6_path = output_dir / "fig6_spearman_corr_and_outlier_lift.png"
    fig6.savefig(fig6_path)
    plt.close(fig6)
    fig_paths["fig6_corr_outlier"] = str(fig6_path)

    return fig_paths


# ==============================================================================
# 7. HÀM CHẠY TRỌN GÓI BƯỚC 1 (PIPELINE ENTRYPOINT + KIỂM THỬ CHỐNG LEAKAGE)
# ==============================================================================
def run_step1_advanced_eda(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    output_dir: str = "./artifacts_step1_eda",
) -> Dict[str, object]:
    """
    Thực thi toàn bộ Bước 1 (Advanced EDA) chỉ trên tập Temporal Train:
    - Kiểm tra bằng assert không có dòng Validation nào được đưa vào tính toán.
    - Xuất đầy đủ các bảng CSV và 3 biểu đồ PNG chuẩn bị cho Bước 2 (Feature Engineering).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_cols, cat_cols = get_ieee_cis_canonical_schema()

    # Chốt chặn chống Data Leakage: Chỉ lấy các dòng thuộc Temporal Train
    assert len(df) == len(split_df), "Kích thước df và split_df không khớp!"
    train_mask = (split_df["temporal_role"].to_numpy() == "train")
    val_mask = (split_df["temporal_role"].to_numpy() == "val")
    train_only_df = df.loc[train_mask].copy()

    # Kiểm chứng ranh giới thời gian
    max_train_dt = float(train_only_df["TransactionDT"].max())
    min_val_dt = float(df.loc[val_mask, "TransactionDT"].min())
    assert max_train_dt < min_val_dt, "LỖI LEAKAGE: Tồn tại dòng trong Temporal Train có TransactionDT >= Temporal Val!"

    # 1. Phân tích Độ phủ Identity, Khuyết có điều kiện + FDR và Gom cụm họ V
    identity_coverage_df, missing_audit_df, v_clusters_df = analyze_conditional_missingness(
        train_only_df, num_cols=num_cols, cat_cols=cat_cols
    )

    # 2. Phân tích Biến Số theo lớp, Ngoại lai P95/P99 và Quan hệ D-Time
    num_dist_df, outlier_df, d_drift_df = analyze_numerical_and_outliers(train_only_df)

    # 3. Phân tích Biến Phân loại (Cramér's V & Chi tiết từng nhóm kèm Count + CI 95%)
    cramers_df, cat_levels_df = analyze_categorical_features(train_only_df, cat_cols=cat_cols)

    # 4. Phân tích Tương quan Hạng Spearman (Pairwise-Complete trước khi điền khuyết)
    target_corr_df, high_collinear_df = analyze_spearman_correlations(train_only_df, num_cols=num_cols)

    # 5. Xuất biểu đồ và lưu bảng kết quả
    fig_paths = generate_step1_figures(
        train_only_df,
        identity_coverage_df=identity_coverage_df,
        cat_levels_df=cat_levels_df,
        target_corr_df=target_corr_df,
        outlier_df=outlier_df,
        output_dir=out_path,
    )

    identity_coverage_df.to_csv(out_path / "eda_identity_coverage.csv", index=False)
    missing_audit_df.to_csv(out_path / "eda_conditional_missingness_fdr.csv", index=False)
    v_clusters_df.drop(columns=["All_Columns"]).to_csv(out_path / "eda_v_missing_clusters.csv", index=False)
    num_dist_df.to_csv(out_path / "eda_numerical_by_class.csv", index=False)
    outlier_df.to_csv(out_path / "eda_outliers_p95_p99.csv", index=False)
    d_drift_df.to_csv(out_path / "eda_d_features_time_drift.csv", index=False)
    cramers_df.to_csv(out_path / "eda_categorical_cramers_v.csv", index=False)
    cat_levels_df.to_csv(out_path / "eda_categorical_levels_detail.csv", index=False)
    target_corr_df.to_csv(out_path / "eda_spearman_with_target.csv", index=False)
    high_collinear_df.to_csv(out_path / "eda_high_collinear_pairs_cd.csv", index=False)

    print("=" * 78)
    print("HOÀN TẤT BƯỚC 1: ADVANCED EDA TRÊN TẬP TEMPORAL TRAIN (0% VALIDATION LEAKAGE)")
    print("=" * 78)
    print(f"- Số dòng Temporal Train đưa vào phân tích: {len(train_only_df):,} dòng (Tỷ lệ Fraud: {train_only_df['isFraud'].mean()*100:.2f}%)")
    print(f"- Số cụm mặt nạ khuyết thực tế của V1-V339: {len(v_clusters_df)} cụm")
    print(f"- Số biến có tín hiệu khuyết đáng kể (FDR < 0.05 & |Lift| mạnh): {int(missing_audit_df['Has_Strong_Missing_Signal'].sum())} biến")
    print("\n[Bảng 1.1] So sánh tỷ lệ gian lận theo Độ phủ bảng Identity (has_identity):")
    print(identity_coverage_df.to_string(index=False))
    print("\n[Bảng 1.2] Top 8 biến Phân loại có hệ số Cramér's V (hiệu chỉnh thiên lệch) cao nhất:")
    print(cramers_df.head(8).to_string(index=False))
    print("\n[Bảng 1.3] Phân tích Ngoại lai ở ngưỡng P95 & P99 cho các biến tiêu biểu:")
    print(outlier_df.head(8).to_string(index=False))
    print(f"\nĐã lưu 10 bảng thống kê CSV và 3 biểu đồ PNG tại thư mục: {out_path}")

    return {
        "identity_coverage_df": identity_coverage_df,
        "missing_audit_df": missing_audit_df,
        "v_clusters_df": v_clusters_df,
        "num_dist_df": num_dist_df,
        "outlier_df": outlier_df,
        "d_drift_df": d_drift_df,
        "cramers_df": cramers_df,
        "cat_levels_df": cat_levels_df,
        "target_corr_df": target_corr_df,
        "high_collinear_df": high_collinear_df,
        "fig_paths": fig_paths,
    }


if __name__ == "__main__":
    # Kiểm thử tự động kết nối từ Bước 0 sang Bước 1
    sample_df = _build_synthetic_ieee_cis_sample(n_rows=2000, seed=42)
    # Tạo thêm 1 cặp đa cộng tuyến mẫu C2 ≈ C1 để kiểm thử phát hiện tương quan cao
    sample_df["C2"] = sample_df["C1"] * 1.02
    # Tạo cụm mặt nạ khuyết mẫu cho V1..V10 giống nhau
    v_mask_sample = sample_df["V1"].isna()
    for idx_v in range(2, 11):
        sample_df.loc[v_mask_sample, f"V{idx_v}"] = np.nan
        sample_df.loc[~v_mask_sample, f"V{idx_v}"] = sample_df.loc[~v_mask_sample, "V1"] * (1.0 + 0.01 * idx_v)

    step0_res = run_step0_pipeline(sample_df, output_dir="/tmp/test_step0_artifacts")

    eda_res = run_step1_advanced_eda(
        df=step0_res["df"],
        split_df=step0_res["split_df"],
        output_dir="/tmp/test_step1_eda_artifacts",
    )

    # Kiểm thử bất biến khi hoán vị nhãn Validation (Validation Label Isolation Test cho EDA)
    perturbed_df = step0_res["df"].copy()
    val_mask = (step0_res["split_df"]["temporal_role"] == "val")
    perturbed_df.loc[val_mask, "isFraud"] = 1 - perturbed_df.loc[val_mask, "isFraud"]

    eda_res_perturbed = run_step1_advanced_eda(
        df=perturbed_df,
        split_df=step0_res["split_df"],
        output_dir="/tmp/test_step1_eda_perturbed",
    )

    assert eda_res["identity_coverage_df"].equals(eda_res_perturbed["identity_coverage_df"]), "Lỗi Leakage ở Identity Coverage!"
    assert eda_res["cramers_df"].equals(eda_res_perturbed["cramers_df"]), "Lỗi Leakage ở Cramér's V!"
    assert eda_res["target_corr_df"].equals(eda_res_perturbed["target_corr_df"]), "Lỗi Leakage ở Spearman Correlation!"
    assert len(eda_res["high_collinear_df"]) >= 1, "Không phát hiện được cặp đa cộng tuyến C1-C2!"
    print("\n[SELF-TEST PASSED] Bước 1 hoàn toàn độc lập với nhãn Validation (100% Leak-Free)!")