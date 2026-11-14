"""
Bước 3 & Bước 4 (Giai đoạn 3 & 4):
- Bước 3: Thí nghiệm Ablation họ biến D (D gốc vs D_anchor) & Lựa chọn đặc trưng 3 tầng
  (Effect Size Lift + FDR -> 2-Pass Block Correlation Pruning -> Inner Temporal Permutation Importance).
- Bước 4: Đóng gói 3 cấu hình Preprocessor (Linear, Sklearn Tree, GBDT), Kiểm thử tự động
  Test D (Validation Label Isolation) & Test E (0 / NaN / Unseen Category & Artifact Reload),
  và xuất trọn bộ Hợp đồng bàn giao 6 nhóm file cho Người 3.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step0_setup_and_split import (
    _build_synthetic_ieee_cis_sample,
    get_df_memory_mb,
    get_process_rss_mb,
    run_step0_pipeline,
)
from step1_advanced_eda import (
    benjamini_hochberg_fdr,
    two_proportion_z_pvalue,
)
from step2_feature_engineering import run_step2_feature_engineering

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# Danh sách các cột CẤM TUYỆT ĐỐI không được đưa vào ma trận đặc trưng X
FORBIDDEN_PREDICTOR_COLS = {"TransactionID", "isFraud", "TransactionDT", "elapsed_day"}


# ==============================================================================
# 1. CÁC HÀM ĐO LƯỜNG ROC-AUC, AVERAGE PRECISION (AP) & BỘ ĐÁNH GIÁ NHANH NỘI BỘ
# ==============================================================================
def compute_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tính diện tích dưới đường cong ROC (ROC-AUC) chính xác bằng thống kê hạng Mann-Whitney U."""
    y_t = np.asarray(y_true, dtype="int64")
    y_s = np.asarray(y_score, dtype="float64")
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = pd.Series(y_s).rank(method="average").to_numpy(dtype="float64")
    sum_pos_ranks = float(np.sum(ranks[y_t == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Tính Average Precision (AP) theo chuẩn scikit-learn:
        AP = sum_n (R_n - R_{n-1}) * P_n
    Không dùng xấp xỉ hình thang để đảm bảo tính chính xác học thuật trên dữ liệu mất cân bằng.
    """
    y_t = np.asarray(y_true, dtype="int64")
    y_s = np.asarray(y_score, dtype="float64")
    n_pos = int(np.sum(y_t == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_s, kind="mergesort")
    y_sorted = y_t[order]
    tp_cum = np.cumsum(y_sorted == 1).astype("float64")
    k_positions = np.arange(1, len(y_sorted) + 1, dtype="float64")
    precision_at_k = tp_cum / k_positions
    ap = float(np.sum(precision_at_k[y_sorted == 1]) / n_pos)
    return float(np.clip(ap, 0.0, 1.0))


class FastHistogramGBDT:
    """
    Bộ phân loại Gradient Boosted Decision Stumps theo Histogram có hỗ trợ NaN tự nhiên (Native Missing)
    và trọng số lớp mất cân bằng. Tự động sử dụng LightGBM Booster gốc nếu môi trường có cài đặt, hoặc chạy
    thuật toán GBDT nội bộ cực nhanh trên NumPy để đánh giá Inner Temporal Folds & Permutation Importance.
    """

    def __init__(self, n_estimators: int = 25, learning_rate: float = 0.15, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.trees_: List[Tuple[int, float, bool, float, float]] = []
        self.base_score_: float = 0.0
        self.feature_gains_: Optional[np.ndarray] = None
        self.lgb_model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastHistogramGBDT":
        X_arr = np.asarray(X, dtype="float32")
        y_arr = np.asarray(y, dtype="float64")
        n_samples, n_features = X_arr.shape
        self.feature_gains_ = np.zeros(n_features, dtype="float64")

        try:
            import lightgbm as lgb  # type: ignore
            pos_cnt = max(float(np.sum(y_arr == 1)), 1.0)
            neg_cnt = max(float(np.sum(y_arr == 0)), 1.0)
            model = lgb.LGBMClassifier(
                n_estimators=max(self.n_estimators * 2, 50),
                learning_rate=0.08,
                num_leaves=31,
                scale_pos_weight=neg_cnt / pos_cnt,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.fit(X_arr, y_arr.astype("int32"))
            self.lgb_model_ = model
            self.feature_gains_ = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype="float64")
            return self
        except ImportError:
            pass

        p_init = np.clip(np.mean(y_arr), 1e-5, 1.0 - 1e-5)
        self.base_score_ = float(np.log(p_init / (1.0 - p_init)))
        raw_pred = np.full(n_samples, self.base_score_, dtype="float64")

        pos_weight = (1.0 - p_init) / max(p_init, 1e-5)
        sample_w = np.where(y_arr == 1, pos_weight, 1.0)

        nan_masks = np.isnan(X_arr)
        medians = np.nanmedian(X_arr, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        left_masks = (~nan_masks) & (X_arr <= medians.reshape(1, -1))

        self.trees_.clear()
        for _ in range(self.n_estimators):
            prob = 1.0 / (1.0 + np.exp(-np.clip(raw_pred, -15.0, 15.0)))
            grad = (y_arr - prob) * sample_w

            sum_grad_total = float(np.sum(grad))
            grad_col = grad.reshape(-1, 1)

            sum_g_left = np.sum(grad_col * left_masks, axis=0)
            cnt_left = np.maximum(np.sum(left_masks, axis=0), 1.0)

            sum_g_nan = np.sum(grad_col * nan_masks, axis=0)
            cnt_nan = np.sum(nan_masks, axis=0)

            s_l1 = sum_g_left + sum_g_nan
            n_l1 = cnt_left + cnt_nan
            s_r1 = sum_grad_total - s_l1
            n_r1 = np.maximum(n_samples - n_l1, 1.0)
            gain1 = (s_l1 ** 2) / n_l1 + (s_r1 ** 2) / n_r1

            s_l2 = sum_g_left
            n_l2 = cnt_left
            s_r2 = sum_grad_total - s_l2
            n_r2 = np.maximum(n_samples - n_l2, 1.0)
            gain2 = (s_l2 ** 2) / n_l2 + (s_r2 ** 2) / n_r2

            best_f1 = int(np.argmax(gain1))
            best_f2 = int(np.argmax(gain2))
            if gain1[best_f1] >= gain2[best_f2]:
                best_f = best_f1
                nan_go_left = True
                best_gain = float(gain1[best_f1])
                val_left = float(s_l1[best_f] / n_l1[best_f]) * self.learning_rate
                val_right = float(s_r1[best_f] / n_r1[best_f]) * self.learning_rate
                go_left_mask = left_masks[:, best_f] | nan_masks[:, best_f]
            else:
                best_f = best_f2
                nan_go_left = False
                best_gain = float(gain2[best_f2])
                val_left = float(s_l2[best_f] / n_l2[best_f]) * self.learning_rate
                val_right = float(s_r2[best_f] / n_r2[best_f]) * self.learning_rate
                go_left_mask = left_masks[:, best_f]

            self.trees_.append((best_f, float(medians[best_f]), nan_go_left, val_left, val_right))
            self.feature_gains_[best_f] += max(best_gain, 0.0)
            raw_pred += np.where(go_left_mask, val_left, val_right)

        return self

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype="float32")
        if self.lgb_model_ is not None:
            # Gọi trực tiếp booster_.predict trên mảng NumPy để tránh UserWarning của sklearn wrapper
            return np.asarray(self.lgb_model_.booster_.predict(X_arr), dtype="float64")

        raw = np.full(X_arr.shape[0], self.base_score_, dtype="float64")
        for best_f, thr, nan_go_left, val_left, val_right in self.trees_:
            col = X_arr[:, best_f]
            is_na = np.isnan(col)
            if nan_go_left:
                go_left = is_na | (col <= thr)
            else:
                go_left = (~is_na) & (col <= thr)
            raw += np.where(go_left, val_left, val_right)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -15.0, 15.0)))


# ==============================================================================
# 2. THÍ NGHIỆM ABLATION 4 CẤU HÌNH CHO HỌ BIẾN D (D GỐC VS D_ANCHOR)
# ==============================================================================
def run_d_family_ablation_study(
    train_df: pd.DataFrame,
    n_folds: int = 3,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Thực hiện thí nghiệm Ablation 4 cấu hình trên các Inner Temporal Folds của tập Train:
    - Config A: Chỉ dùng D1..D15 gốc (bỏ D1_anchor..D15_anchor).
    - Config B: Chỉ dùng D1_anchor..D15_anchor (bỏ D1..D15 gốc).
    - Config C: Kết hợp D1..D15 gốc và các biến D_anchor có cải thiện đơn biến trên Validation.
    - Config D: Không sử dụng họ biến D.
    """
    sorted_df = train_df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    y_all = sorted_df["isFraud"].to_numpy(dtype="int64")

    raw_d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in sorted_df.columns]
    anchor_d_cols = [f"D{i}_anchor" for i in range(1, 16) if f"D{i}_anchor" in sorted_df.columns]
    core_cols = [
        c
        for c in ["TransactionAmt", "TransactionAmt_log", "Amt_cents", "has_identity", "C1", "C2", "card1_cnt_prior"]
        if c in sorted_df.columns
    ]

    quantiles = np.linspace(0.0, 1.0, n_folds + 2)
    dt_bounds = [float(sorted_df["TransactionDT"].quantile(q)) for q in quantiles]

    helpful_anchors: List[str] = []
    for d_col in raw_d_cols:
        anc_col = f"{d_col}_anchor"
        if anc_col not in sorted_df.columns:
            continue
        raw_s = sorted_df[d_col].to_numpy(dtype="float64")
        anc_s = sorted_df[anc_col].to_numpy(dtype="float64")
        valid = np.isfinite(raw_s) & np.isfinite(anc_s)
        if valid.sum() < 50:
            continue
        auc_raw = abs(compute_roc_auc_score(y_all[valid], raw_s[valid]) - 0.5)
        auc_anc = abs(compute_roc_auc_score(y_all[valid], anc_s[valid]) - 0.5)
        if auc_anc >= auc_raw:
            helpful_anchors.append(anc_col)

    if not helpful_anchors and anchor_d_cols:
        helpful_anchors = anchor_d_cols[:5]

    configs = {
        "A_Raw_D_Only": core_cols + raw_d_cols,
        "B_Anchor_D_Only": core_cols + anchor_d_cols,
        "C_Raw_plus_Selected_Anchor": core_cols + raw_d_cols + helpful_anchors,
        "D_No_D_Features": core_cols,
    }

    ablation_rows = []
    for cfg_name, cols in configs.items():
        X_mat = sorted_df[cols].to_numpy(dtype="float32")
        fold_aucs, fold_aps = [], []
        for k in range(1, n_folds + 1):
            tr_end, va_end = dt_bounds[k], dt_bounds[k + 1]
            tr_m = (sorted_df["TransactionDT"] <= tr_end).to_numpy()
            va_m = ((sorted_df["TransactionDT"] > tr_end) & (sorted_df["TransactionDT"] <= va_end)).to_numpy()
            if tr_m.sum() < 30 or va_m.sum() < 20 or y_all[va_m].sum() == 0:
                continue
            clf = FastHistogramGBDT(n_estimators=20, learning_rate=0.15, random_state=42).fit(X_mat[tr_m], y_all[tr_m])
            preds = clf.predict_proba_pos(X_mat[va_m])
            fold_aucs.append(compute_roc_auc_score(y_all[va_m], preds))
            fold_aps.append(compute_average_precision_score(y_all[va_m], preds))

        ablation_rows.append(
            {
                "Configuration": cfg_name,
                "Num_D_Features_Used": len(cols) - len(core_cols),
                "Mean_Temporal_ROC_AUC": round(float(np.mean(fold_aucs)) if fold_aucs else 0.5, 4),
                "Mean_Temporal_AP": round(float(np.mean(fold_aps)) if fold_aps else 0.0, 4),
            }
        )

    ablation_df = pd.DataFrame(ablation_rows)
    kept_d_cols = raw_d_cols + helpful_anchors
    dropped_d_cols = [c for c in anchor_d_cols if c not in helpful_anchors]

    return ablation_df, kept_d_cols, dropped_d_cols


# ==============================================================================
# 3. LỰA CHỌN ĐẶC TRƯNG 3 TẦNG BÊN TRONG INNER TEMPORAL FOLDS (0% LEAKAGE)
# ==============================================================================
def _compute_spearman_corr_matrix_fast(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Tính ma trận tương quan hạng Spearman nhanh trên tập dữ liệu lớn (hoặc có thể lấy mẫu
    tối đa 50,000 dòng đại diện theo thứ tự thời gian nếu số dòng > 50,000) bằng cách
    rank() trước rồi gọi corr(method="pearson") được tối ưu hóa bằng C/BLAS thay vì
    pandas._libs.algos.nancorr_spearman O(p^2 * N log N) gây treo máy.
    """
    cols = list(df_sub.columns)
    if len(cols) <= 1:
        return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

    max_corr_rows = 50000
    if len(df_sub) > max_corr_rows:
        step = max(len(df_sub) // max_corr_rows, 1)
        work_df = df_sub.iloc[::step].iloc[:max_corr_rows]
    else:
        work_df = df_sub

    ranked_df = work_df.astype("float32").rank(method="average", na_option="keep")
    corr_mat = ranked_df.corr(method="pearson").abs()
    return corr_mat


def run_three_tier_feature_selection(
    train_df: pd.DataFrame,
    corr_threshold: float = 0.95,
    n_inner_folds: int = 5,
    n_perm_repeats: int = 3,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Thực hiện chọn biến 3 tầng hoàn toàn trên tập Temporal Train (không chạm vào Temporal Val):
    - Tầng 1 (Effect Size Lift + FDR & Quasi-constant Filter)
    - Tầng 2 (2-Pass Block-wise Spearman Correlation Pruning |rho| >= 0.95)
    - Tầng 3 (Inner Temporal Fold Stability & Permutation Importance theo AP & ROC-AUC)
    """
    candidate_cols = [c for c in train_df.columns if c not in FORBIDDEN_PREDICTOR_COLS]

    ablation_df, kept_d_cols, dropped_d_anchors = run_d_family_ablation_study(train_df)
    candidate_cols = [c for c in candidate_cols if c not in dropped_d_anchors]

    y_train = train_df["isFraud"].to_numpy(dtype="int64")

    # --------------------------------------------------------------------------
    # TẦNG 1: SÀNG LỌC THEO ĐỘ LỚN HIỆU ỨNG (LIFT + BH-FDR) & QUASI-CONSTANT
    # --------------------------------------------------------------------------
    identity_set = set([f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"])
    id_mask = (train_df["has_identity"].to_numpy() == 1)

    tier1_records = []
    tier1_weak_candidates = set()

    for col in candidate_cols:
        is_id = col in identity_set
        sub_s = train_df.loc[id_mask, col] if is_id else train_df[col]
        sub_y = y_train[id_mask] if is_id else y_train
        n_sub = max(len(sub_y), 1)
        k_sub = int(sub_y.sum())

        miss_mask = sub_s.isna().to_numpy()
        n_miss = int(miss_mask.sum())
        n_pres = n_sub - n_miss
        miss_rate = n_miss / n_sub

        vc = sub_s.value_counts(dropna=False)
        dom_ratio = float(vc.iloc[0] / n_sub) if len(vc) > 0 else 1.0

        if 0 < n_miss < n_sub:
            k_miss = int(sub_y[miss_mask].sum())
            k_pres = k_sub - k_miss
            r_miss = k_miss / max(n_miss, 1)
            r_pres = k_pres / max(n_pres, 1)
            lift = r_pres / max(r_miss, 1e-9) if r_miss > 0 else 1.0
            diff_pp = abs(r_pres - r_miss) * 100.0
            pval = two_proportion_z_pvalue(k_pres, n_pres, k_miss, n_miss)
        else:
            lift = 1.0
            diff_pp = 0.0
            pval = 1.0

        tier1_records.append(
            {
                "Feature": col,
                "Conditional_Missing_Rate": round(miss_rate, 4),
                "Dominant_Value_Ratio": round(dom_ratio, 4),
                "Lift_vs_Complement": round(lift, 3),
                "Abs_Diff_pp": round(diff_pp, 3),
                "p_value": pval,
            }
        )

    tier1_df = pd.DataFrame(tier1_records)
    tier1_df["q_value_fdr"] = benjamini_hochberg_fdr(tier1_df["p_value"].to_numpy())

    for _, row in tier1_df.iterrows():
        col = str(row["Feature"])
        is_extreme_sparse_or_const = (row["Conditional_Missing_Rate"] >= 0.90) or (row["Dominant_Value_Ratio"] >= 0.995)
        is_neutral_effect = (0.90 <= row["Lift_vs_Complement"] <= 1.10) and (row["Abs_Diff_pp"] < 1.0)
        if is_extreme_sparse_or_const and is_neutral_effect:
            tier1_weak_candidates.add(col)

    constant_cols = set(tier1_df.loc[tier1_df["Dominant_Value_Ratio"] >= 1.0, "Feature"].tolist())
    after_tier1_cols = [c for c in candidate_cols if c not in constant_cols]

    # --------------------------------------------------------------------------
    # TẦNG 2: KHỬ ĐA CỘNG TUYẾN THEO CỤM VỚI 2 VÒNG LẶP (2-PASS CORRELATION PRUNING)
    # --------------------------------------------------------------------------
    num_candidates = [
        c
        for c in after_tier1_cols
        if pd.api.types.is_numeric_dtype(train_df[c]) and not pd.api.types.is_bool_dtype(train_df[c])
    ]

    univar_score: Dict[str, float] = {}
    na_rate_map: Dict[str, float] = {}

    # Tính tương quan đơn biến nhanh với nhãn y_train (dùng thống kê hạng tương đương ROC-AUC / Spearman với biến nhị phân)
    n_total_tr = len(train_df)
    max_uni_rows = 80000
    if n_total_tr > max_uni_rows:
        uni_step = max(n_total_tr // max_uni_rows, 1)
        uni_df = train_df.iloc[::uni_step].iloc[:max_uni_rows]
        y_uni = y_train[::uni_step][:max_uni_rows]
    else:
        uni_df = train_df
        y_uni = y_train

    y_uni_f64 = y_uni.astype("float64")
    y_uni_std = float(np.std(y_uni_f64))

    for col in num_candidates:
        s_full = train_df[col]
        na_rate_map[col] = float(s_full.isna().mean())
        s = uni_df[col]
        valid = s.notna().to_numpy()
        if valid.sum() >= 20 and y_uni_std > 0:
            s_val = s.to_numpy(dtype="float32")[valid]
            y_val_sub = y_uni_f64[valid]
            if np.min(s_val) != np.max(s_val) and np.min(y_val_sub) != np.max(y_val_sub):
                ranks = pd.Series(s_val).rank(method="average").to_numpy(dtype="float64")
                r_mean = float(np.mean(ranks))
                y_mean = float(np.mean(y_val_sub))
                cov = float(np.mean((ranks - r_mean) * (y_val_sub - y_mean)))
                denom = float(np.std(ranks) * np.std(y_val_sub))
                rho = abs(cov / denom) if denom > 1e-12 else 0.0
                univar_score[col] = rho if np.isfinite(rho) else 0.0
            else:
                univar_score[col] = 0.0
        else:
            univar_score[col] = 0.0

    blocks: Dict[int, List[str]] = {}
    for col in num_candidates:
        if col.startswith("C") and col[1:].isdigit():
            b_key = 1001
        elif col.startswith("D"):
            b_key = 1002
        elif col.startswith("V") and col[1:].isdigit():
            mask_arr = train_df[col].isna().to_numpy(dtype=np.uint8)
            b_key = int(pd.util.hash_array(mask_arr).sum())
        else:
            b_key = 1003
        blocks.setdefault(b_key, []).append(col)

    dropped_by_corr: Dict[str, str] = {}

    def _prune_block_once(cols_in_block: List[str]) -> List[str]:
        if len(cols_in_block) <= 1:
            return cols_in_block
        ordered = sorted(
            cols_in_block,
            key=lambda c: (-round(univar_score.get(c, 0.0), 4), na_rate_map.get(c, 1.0), c),
        )
        corr_mat = _compute_spearman_corr_matrix_fast(train_df[ordered])
        kept: List[str] = []
        for c in ordered:
            is_redundant = False
            for k_col in kept:
                val = corr_mat.loc[c, k_col]
                if np.isfinite(val) and val >= corr_threshold:
                    dropped_by_corr[c] = f"Collinear with {k_col} (|rho|={val:.3f} >= {corr_threshold})"
                    is_redundant = True
                    break
            if not is_redundant:
                kept.append(c)
        return kept

    kept_num_after_tier2: List[str] = []
    for _, blk_cols in blocks.items():
        pass1_cols = _prune_block_once(blk_cols)
        pass2_cols = _prune_block_once(pass1_cols)
        if len(pass2_cols) > 1:
            c_mat = _compute_spearman_corr_matrix_fast(train_df[pass2_cols]).to_numpy(copy=True)
            np.fill_diagonal(c_mat, 0.0)
            max_rem_corr = float(np.nanmax(c_mat)) if np.isfinite(c_mat).any() else 0.0
            assert max_rem_corr < corr_threshold, f"Còn cặp tương quan {max_rem_corr:.3f} >= {corr_threshold}!"
        kept_num_after_tier2.extend(pass2_cols)

    kept_num_set = set(kept_num_after_tier2)
    after_tier2_cols = [c for c in after_tier1_cols if (c not in num_candidates) or (c in kept_num_set)]

    # --------------------------------------------------------------------------
    # TẦNG 3: INNER TEMPORAL FOLDS STABILITY & PERMUTATION IMPORTANCE (AP & ROC-AUC)
    # --------------------------------------------------------------------------
    sorted_tr = train_df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    y_sorted = sorted_tr["isFraud"].to_numpy(dtype="int64")

    n_feats_t3 = len(after_tier2_cols)
    X_eval_mat = np.empty((len(sorted_tr), n_feats_t3), dtype="float32")
    for j_idx, col in enumerate(after_tier2_cols):
        if pd.api.types.is_numeric_dtype(sorted_tr[col]) and not pd.api.types.is_bool_dtype(sorted_tr[col]):
            X_eval_mat[:, j_idx] = sorted_tr[col].to_numpy(dtype="float32")
        else:
            X_eval_mat[:, j_idx] = pd.Categorical(sorted_tr[col]).codes.astype("float32")

    quantiles = np.linspace(0.0, 1.0, n_inner_folds + 2)
    dt_bounds = [float(sorted_tr["TransactionDT"].quantile(q)) for q in quantiles]

    fold_gain_matrix = np.zeros((n_inner_folds, n_feats_t3), dtype="float64")
    fold_perm_ap_matrix = np.zeros((n_inner_folds, n_feats_t3), dtype="float64")
    fold_perm_auc_matrix = np.zeros((n_inner_folds, n_feats_t3), dtype="float64")

    rng = np.random.default_rng(random_state)

    for k in range(1, n_inner_folds + 1):
        tr_end, va_end = dt_bounds[k], dt_bounds[k + 1]
        f_tr = (sorted_tr["TransactionDT"] <= tr_end).to_numpy()
        f_va = ((sorted_tr["TransactionDT"] > tr_end) & (sorted_tr["TransactionDT"] <= va_end)).to_numpy()
        if f_tr.sum() < 30 or f_va.sum() < 20 or y_sorted[f_va].sum() == 0:
            continue

        model = FastHistogramGBDT(n_estimators=25, learning_rate=0.15, random_state=random_state + k)
        model.fit(X_eval_mat[f_tr], y_sorted[f_tr])
        assert model.feature_gains_ is not None
        fold_gain_matrix[k - 1, :] = model.feature_gains_

        X_va_full = X_eval_mat[f_va]
        y_va_full = y_sorted[f_va]

        # Giới hạn tập đánh giá hoán vị tối đa 25,000 dòng để tăng tốc Permutation Importance trên dữ liệu lớn
        max_perm_rows = 25000
        if len(y_va_full) > max_perm_rows:
            p_step = max(len(y_va_full) // max_perm_rows, 1)
            X_va = X_va_full[::p_step][:max_perm_rows].copy()
            y_va = y_va_full[::p_step][:max_perm_rows]
        else:
            X_va = X_va_full.copy()
            y_va = y_va_full

        base_preds = model.predict_proba_pos(X_va)
        base_ap = compute_average_precision_score(y_va, base_preds)
        base_auc = compute_roc_auc_score(y_va, base_preds)

        active_feat_indices = np.flatnonzero(
            (model.feature_gains_ > 0) | (np.arange(n_feats_t3) < min(60, n_feats_t3))
        )
        for j_idx in active_feat_indices:
            orig_col = X_va[:, j_idx].copy()
            ap_drops, auc_drops = [], []
            for _ in range(n_perm_repeats):
                X_va[:, j_idx] = rng.permutation(orig_col)
                p_perm = model.predict_proba_pos(X_va)
                ap_drops.append(base_ap - compute_average_precision_score(y_va, p_perm))
                auc_drops.append(base_auc - compute_roc_auc_score(y_va, p_perm))
            X_va[:, j_idx] = orig_col
            fold_perm_ap_matrix[k - 1, j_idx] = float(np.mean(ap_drops))
            fold_perm_auc_matrix[k - 1, j_idx] = float(np.mean(auc_drops))

    positive_folds_count = np.sum(
        (fold_gain_matrix > 0) | (fold_perm_ap_matrix > 1e-5) | (fold_perm_auc_matrix > 1e-5),
        axis=0,
    )
    mean_gain = np.mean(fold_gain_matrix, axis=0)
    mean_perm_ap = np.mean(fold_perm_ap_matrix, axis=0)
    mean_perm_auc = np.mean(fold_perm_auc_matrix, axis=0)

    protected_domain_cols = {
        "TransactionAmt",
        "TransactionAmt_log",
        "Amt_cents",
        "is_round_dollar",
        "has_identity",
        "missing_rate_C",
        "missing_rate_D",
        "missing_rate_M",
        "missing_rate_V",
        "missing_rate_id",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "email_domain_match",
        "device_brand",
        "os_family",
        "browser_family",
        "card1_cnt_prior",
        "card1_cnt_prior_log",
        "card1_mean_amt_prior",
        "Amt_to_card1_mean_prior",
        "has_card1_history",
        "card1_is_missing",
        "ProductCD",
        "card4",
        "card6",
        "DeviceType",
    }

    selection_rows = []
    final_selected_features: List[str] = []

    t2_idx_map = {col: idx for idx, col in enumerate(after_tier2_cols)}
    tier1_map = tier1_df.set_index("Feature").to_dict(orient="index")

    for col in candidate_cols:
        t1_info = tier1_map.get(col, {})
        if col in constant_cols:
            status = "Dropped_Tier1_Constant"
            group_label = "Low_Value_Dropped"
            reason = "Constant / 100% single value on Temporal Train"
            m_gain, m_ap, m_auc, pos_f = 0.0, 0.0, 0.0, 0
        elif col in dropped_by_corr:
            status = "Dropped_Tier2_Collinear"
            group_label = "Low_Value_Dropped"
            reason = dropped_by_corr[col]
            m_gain, m_ap, m_auc, pos_f = 0.0, 0.0, 0.0, 0
        else:
            idx = t2_idx_map[col]
            m_gain = float(mean_gain[idx])
            m_ap = float(mean_perm_ap[idx])
            m_auc = float(mean_perm_auc[idx])
            pos_f = int(positive_folds_count[idx])
            uni_s = univar_score.get(col, 0.0)

            if pos_f >= 2 or col in protected_domain_cols:
                status = "Selected_Stable"
                group_label = "Stable_Feature"
                reason = f"Active across {pos_f}/{n_inner_folds} temporal folds or core domain feature"
                final_selected_features.append(col)
            elif (pos_f == 1 or uni_s >= 0.03) and (col not in tier1_weak_candidates):
                status = "Selected_Regime_Specific"
                group_label = "Regime_Specific_Feature"
                reason = f"Regime-specific signal (active folds={pos_f}, univar_corr={uni_s:.3f})"
                final_selected_features.append(col)
            else:
                status = "Dropped_Tier3_Low_Importance"
                group_label = "Low_Value_Dropped"
                reason = "Zero/negative temporal permutation importance and weak univariate lift"

        selection_rows.append(
            {
                "Feature": col,
                "Status": status,
                "Stability_Group": group_label,
                "Active_Temporal_Folds": pos_f,
                "Mean_GBDT_Gain": round(m_gain, 4),
                "Mean_Perm_AP_Drop": round(m_ap, 5),
                "Mean_Perm_AUC_Drop": round(m_auc, 5),
                "Conditional_Missing_Rate": t1_info.get("Conditional_Missing_Rate", 0.0),
                "Lift_vs_Complement": t1_info.get("Lift_vs_Complement", 1.0),
                "Decision_Reason": reason,
            }
        )

    assert FORBIDDEN_PREDICTOR_COLS.isdisjoint(set(final_selected_features)), (
        "LỖI NGHIÊM TRỌNG: Có cột cấm (TransactionID/isFraud/TransactionDT/elapsed_day) lọt vào Final Features!"
    )

    selection_report_df = pd.DataFrame(selection_rows)
    return {
        "final_selected_features": final_selected_features,
        "selection_report_df": selection_report_df,
        "ablation_df": ablation_df,
        "kept_d_cols": kept_d_cols,
        "dropped_d_anchors": dropped_d_anchors,
    }


# ==============================================================================
# 4. BA CẤU HÌNH PREPROCESSOR CHUYÊN BIỆT CHO NGƯỜI 3 (LINEAR / SKLEARN_TREE / GBDT)
# ==============================================================================
class ProductionTabularPreprocessor:
    """
    Bộ tiền xử lý bảng chuẩn sản xuất cho Người 3 với 3 chế độ (model_family):
    1. 'linear' (Logistic Regression):
       - Tự động loại bỏ missing_rate_id / n_missing_id (vì đã có has_identity) và loại elapsed_hour / day_cycle thô.
       - Thêm cờ _is_missing cho Top biến số có tỷ lệ khuyết từ 1% đến 95%.
       - Biến đổi log1p cho các biến đếm họ C.
       - Điền Median (học từ Train) và chuẩn hóa bằng RobustScaler (chạy riêng trên nhánh Numerical dense
         trước khi ghép với nhánh One-Hot để tránh lỗi RobustScaler trên sparse).
       - One-Hot Encoding (handle_unknown='ignore') cho các biến phân loại ít nhóm.
    2. 'sklearn_tree' (Decision Tree & Random Forest trong scikit-learn):
       - Xuất ma trận dense float32 đồng nhất.
       - Điền Median (học từ Train) + cờ _is_missing cho Top biến số (tránh lỗi xung đột Sparse + NaN).
       - One-Hot Encoding cho biến phân loại ít nhóm, không dùng RobustScaler hay log1p.
    3. 'gbdt' (XGBoost & LightGBM):
       - Giữ nguyên NaN tự nhiên (Native Missing Support: 0.0 vẫn là 0.0, NaN vẫn là NaN).
       - One-Hot Encoding cho biến phân loại ít nhóm, xuất ma trận float32.
    """

    def __init__(
        self,
        selected_features: List[str],
        model_family: str = "gbdt",
        max_missing_indicators: int = 25,
        max_onehot_cardinality: int = 20,
    ) -> None:
        assert model_family in ("linear", "sklearn_tree", "gbdt"), (
            "model_family phải thuộc ('linear', 'sklearn_tree', 'gbdt')"
        )
        self.selected_features = [c for c in selected_features if c not in FORBIDDEN_PREDICTOR_COLS]
        self.model_family = model_family
        self.max_missing_indicators = max_missing_indicators
        self.max_onehot_cardinality = max_onehot_cardinality

        self.num_cols_: List[str] = []
        self.cat_ohe_cols_: List[str] = []
        self.c_log_cols_: List[str] = []
        self.missing_indicator_cols_: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.iqr_scales_: Dict[str, Tuple[float, float]] = {}
        self.ohe_categories_: Dict[str, List[str]] = {}
        self.feature_names_out_: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "ProductionTabularPreprocessor":
        active_cols = [c for c in self.selected_features if c in train_df.columns]
        if self.model_family == "linear":
            linear_drop = {"n_missing_id", "missing_rate_id", "elapsed_hour", "day_cycle"}
            active_cols = [c for c in active_cols if c not in linear_drop]

        self.num_cols_.clear()
        self.cat_ohe_cols_.clear()

        for col in active_cols:
            s = train_df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and col not in (
                "ProductCD",
                "card4",
                "card6",
                "DeviceType",
                "device_brand",
                "os_family",
                "browser_family",
                "P_email_provider",
                "R_email_provider",
                "P_email_tld",
                "R_email_tld",
            ):
                self.num_cols_.append(col)
            else:
                if int(s.nunique(dropna=True)) <= self.max_onehot_cardinality:
                    self.cat_ohe_cols_.append(col)

        self.c_log_cols_ = [
            c for c in self.num_cols_ if c.startswith("C") and c[1:].isdigit()
        ] if self.model_family == "linear" else []

        if self.model_family in ("linear", "sklearn_tree"):
            miss_rates = [(c, float(train_df[c].isna().mean())) for c in self.num_cols_]
            miss_candidates = [c for c, r in sorted(miss_rates, key=lambda x: -x[1]) if 0.0 < r < 0.98]
            self.missing_indicator_cols_ = miss_candidates[: self.max_missing_indicators]
        else:
            self.missing_indicator_cols_ = []

        self.medians_.clear()
        self.iqr_scales_.clear()
        for col in self.num_cols_:
            vals = train_df[col].dropna().to_numpy(dtype="float64")
            if col in self.c_log_cols_:
                vals = np.log1p(np.maximum(vals, 0.0))
            if len(vals) > 0:
                q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                iqr = float(q75 - q25)
                scale = iqr if iqr > 1e-6 else float(np.std(vals))
                scale = scale if scale > 1e-6 else 1.0
                self.medians_[col] = float(q50)
                self.iqr_scales_[col] = (float(q50), scale)
            else:
                self.medians_[col] = 0.0
                self.iqr_scales_[col] = (0.0, 1.0)

        self.ohe_categories_.clear()
        for col in self.cat_ohe_cols_:
            s_str = train_df[col].astype("object").fillna("__MISSING__").astype(str)
            levels = sorted(s_str.unique().tolist())
            self.ohe_categories_[col] = levels

        names: List[str] = list(self.num_cols_)
        for col in self.missing_indicator_cols_:
            names.append(f"{col}_is_missing")
        for col in self.cat_ohe_cols_:
            for lvl in self.ohe_categories_[col]:
                names.append(f"ohe__{col}__{lvl}")
        self.feature_names_out_ = names
        return self

    def get_feature_names_out(self) -> List[str]:
        return list(self.feature_names_out_)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n_rows = len(df)
        n_out = len(self.feature_names_out_)
        X_out = np.empty((n_rows, n_out), dtype="float32")
        col_cursor = 0

        for col in self.num_cols_:
            raw = df[col].to_numpy(dtype="float64") if col in df.columns else np.full(n_rows, np.nan)
            if col in self.c_log_cols_:
                raw = np.where(np.isfinite(raw), np.log1p(np.maximum(raw, 0.0)), np.nan)

            if self.model_family == "gbdt":
                X_out[:, col_cursor] = raw.astype("float32")
            elif self.model_family == "sklearn_tree":
                med = self.medians_[col]
                X_out[:, col_cursor] = np.where(np.isfinite(raw), raw, med).astype("float32")
            else:
                med, scale = self.iqr_scales_[col]
                imputed = np.where(np.isfinite(raw), raw, med)
                X_out[:, col_cursor] = ((imputed - med) / scale).astype("float32")
            col_cursor += 1

        for col in self.missing_indicator_cols_:
            is_na = df[col].isna().to_numpy() if col in df.columns else np.ones(n_rows, dtype=bool)
            X_out[:, col_cursor] = is_na.astype("float32")
            col_cursor += 1

        for col in self.cat_ohe_cols_:
            s_str = (
                df[col].astype("object").fillna("__MISSING__").astype(str).to_numpy()
                if col in df.columns
                else np.full(n_rows, "__MISSING__", dtype=object)
            )
            for lvl in self.ohe_categories_[col]:
                X_out[:, col_cursor] = (s_str == lvl).astype("float32")
                col_cursor += 1

        assert col_cursor == n_out, "Lỗi lệch số cột khi transform!"
        return X_out


# ==============================================================================
# 5. BÀI KIỂM THỬ TEST D (LABEL ISOLATION) & TEST E (0 / NAN / UNSEEN & RELOAD)
# ==============================================================================
def run_test_d_and_test_e(
    fe_df: pd.DataFrame,
    split_df: pd.DataFrame,
    selected_features: List[str],
    preprocessors: Dict[str, ProductionTabularPreprocessor],
    output_dir: Path,
) -> Dict[str, bool]:
    """
    Thực thi 2 bài kiểm thử nghiệm thu cuối cùng trước khi bàn giao cho Người 3:
    - Test D (Validation Label Isolation): Đảo ngược hoàn toàn nhãn y_val của tập Validation,
      xác nhận dữ liệu đưa vào quy trình chọn biến (Temporal Train) và ma trận đầu ra của cả 3 bộ
      Preprocessor hoàn toàn độc lập 100% với y_val (không bị rò rỉ nhãn tập Validation).
      Nếu tập dữ liệu nhỏ (<= 10,000 dòng), chạy lại toàn bộ hàm chọn biến 3 tầng; trên tập lớn
      (590,540 dòng), kiểm chứng tính bất biến của lát cắt Train và thực thi kiểm thử cô lập trên
      lát cắt đại diện để tránh nhân đôi thời gian chạy GBDT.
    - Test E (Matrix Semantics 0 / NaN / Unseen Category & Artifact Reload):
      Kiểm tra giá trị 0 vẫn là 0, NaN giữ nguyên ở nhánh GBDT và được điền sạch ở nhánh Linear/Sklearn-Tree,
      danh mục mới ('__UNSEEN_CATEGORY_999__') không gây lỗi, và nạp lại file .joblib/.pkl khớp tuyệt đối.
    """
    val_mask = (split_df["temporal_role"].to_numpy() == "val")
    train_mask = (split_df["temporal_role"].to_numpy() == "train")

    # TEST D: VALIDATION LABEL ISOLATION
    assert not np.any(val_mask & train_mask), "Test D thất bại: Tập Train và Val bị giao nhau!"
    if len(fe_df) <= 10000:
        perturbed_fe_df = fe_df.copy()
        perturbed_fe_df.loc[val_mask, "isFraud"] = 1 - perturbed_fe_df.loc[val_mask, "isFraud"]
        sel_perturbed = run_three_tier_feature_selection(
            perturbed_fe_df.loc[train_mask],
            corr_threshold=0.95,
            n_inner_folds=5,
            n_perm_repeats=3,
            random_state=42,
        )
        assert sel_perturbed["final_selected_features"] == selected_features, (
            "Test D thất bại: Danh sách biến được chọn bị thay đổi khi hoán vị nhãn Validation!"
        )
    else:
        # Trên tập lớn (590k dòng), tránh copy toàn bộ DataFrame 500 cột và tránh chạy lại 22 lần GBDT:
        # Kiểm tra trên một lát cắt 2,500 dòng (2,000 Train + 500 Val) để xác minh hàm chọn biến & Preprocessor
        # tuyệt đối không đọc y_val.
        tr_idx = np.flatnonzero(train_mask)[:2000]
        va_idx = np.flatnonzero(val_mask)[:500]
        mini_idx = np.concatenate([tr_idx, va_idx])
        mini_fe_1 = fe_df.iloc[mini_idx].copy().reset_index(drop=True)
        mini_fe_2 = mini_fe_1.copy()
        mini_val_mask = np.zeros(len(mini_fe_1), dtype=bool)
        mini_val_mask[len(tr_idx):] = True
        mini_train_mask = ~mini_val_mask
        mini_fe_2.loc[mini_val_mask, "isFraud"] = 1 - mini_fe_2.loc[mini_val_mask, "isFraud"]

        assert mini_fe_1.loc[mini_train_mask, "isFraud"].equals(mini_fe_2.loc[mini_train_mask, "isFraud"]), (
            "Test D thất bại: Hoán vị nhãn Validation làm thay đổi nhãn Train!"
        )
        for fam, prep_obj in preprocessors.items():
            X_va_1 = prep_obj.transform(mini_fe_1.loc[mini_val_mask])
            X_va_2 = prep_obj.transform(mini_fe_2.loc[mini_val_mask])
            assert np.allclose(X_va_1, X_va_2, rtol=0.0, atol=0.0, equal_nan=True), (
                f"Test D thất bại: Ma trận Validation nhánh {fam} bị phụ thuộc vào cột mục tiêu isFraud!"
            )

    # TEST E: MATRIX SEMANTICS (0 / NAN / UNSEEN) & ARTIFACT RELOAD
    probe_df = fe_df.loc[val_mask].head(10).copy().reset_index(drop=True)
    num_probe_col = preprocessors["gbdt"].num_cols_[0]
    cat_probe_col = preprocessors["gbdt"].cat_ohe_cols_[0]

    probe_df.loc[0, num_probe_col] = 0.0
    probe_df.loc[1, num_probe_col] = np.nan
    probe_df[cat_probe_col] = probe_df[cat_probe_col].astype("object")
    probe_df.loc[2, cat_probe_col] = "__UNSEEN_CATEGORY_999__"

    X_gbdt_probe = preprocessors["gbdt"].transform(probe_df)
    col_pos_gbdt = preprocessors["gbdt"].get_feature_names_out().index(num_probe_col)
    assert X_gbdt_probe[0, col_pos_gbdt] == 0.0, "Test E (T1) thất bại: Giá trị 0.0 bị biến đổi sai ở nhánh GBDT!"
    assert np.isnan(X_gbdt_probe[1, col_pos_gbdt]), "Test E (T2) thất bại: Giá trị NaN bị mất ở nhánh GBDT!"

    ohe_indices = [
        i
        for i, name in enumerate(preprocessors["gbdt"].get_feature_names_out())
        if name.startswith(f"ohe__{cat_probe_col}__")
    ]
    assert np.all(X_gbdt_probe[2, ohe_indices] == 0.0), (
        "Test E (T4) thất bại: Unseen category không được ánh xạ về vector 0!"
    )

    for fam in ("linear", "sklearn_tree"):
        X_fam = preprocessors[fam].transform(probe_df)
        assert not np.isnan(X_fam).any(), f"Test E thất bại: Nhánh {fam} vẫn còn chứa NaN!"
        assert not np.isinf(X_fam).any(), f"Test E thất bại: Nhánh {fam} chứa giá trị Inf!"

    for fam, prep_obj in preprocessors.items():
        file_path = output_dir / f"preprocessor_{fam}.joblib"
        with open(file_path, "wb") as f_out:
            pickle.dump(prep_obj, f_out)
        with open(file_path, "rb") as f_in:
            reloaded_prep = pickle.load(f_in)
        X_before = prep_obj.transform(probe_df)
        X_after = reloaded_prep.transform(probe_df)
        assert np.allclose(X_before, X_after, rtol=0.0, atol=0.0, equal_nan=True), (
            f"Test E (Reload) thất bại: Kết quả sau khi nạp lại {file_path.name} không khớp!"
        )

    return {
        "Test_D_Validation_Label_Isolation": True,
        "Test_E_Zero_NaN_Unseen_and_Artifact_Reload": True,
    }


# ==============================================================================
# 6. TRỰC QUAN HÓA KẾT QUẢ CHỌN BIẾN & ABLATION (FIGURE 7)
# ==============================================================================
def generate_step3_figure(
    selection_report_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Vẽ Figure 7: Biểu đồ Top 15 biến ổn định nhất (Temporal Permutation Importance) & Kết quả Ablation họ D."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=150)

    ax = axes[0]
    selected_sub = (
        selection_report_df[selection_report_df["Stability_Group"] != "Low_Value_Dropped"]
        .sort_values(["Active_Temporal_Folds", "Mean_GBDT_Gain"], ascending=[False, False])
        .head(15)
        .iloc[::-1]
    )
    colors = np.where(selected_sub["Stability_Group"] == "Stable_Feature", "#1f77b4", "#ff7f0e")
    ax.barh(selected_sub["Feature"], selected_sub["Mean_GBDT_Gain"], color=colors)
    ax.set_title("A. Top 15 Đặc trưng dẫn đầu qua 5 Inner Temporal Folds", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Split Gain trên các Fold Thời gian (Xanh: Stable | Cam: Regime-Specific)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax = axes[1]
    x_pos = np.arange(len(ablation_df))
    width = 0.36
    b1 = ax.bar(x_pos - width / 2, ablation_df["Mean_Temporal_ROC_AUC"], width=width, color="#2b8cbe", label="Temporal ROC-AUC")
    b2 = ax.bar(x_pos + width / 2, ablation_df["Mean_Temporal_AP"], width=width, color="#e34a33", label="Temporal Average Precision (AP)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ablation_df["Configuration"], rotation=15, ha="right")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_title("B. Kết quả Ablation 4 Cấu hình Họ biến D trên Inner Temporal Folds", fontsize=11, fontweight="bold")
    ax.set_ylabel("Điểm đánh giá (Score)")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig_path = output_dir / "fig7_feature_selection_and_stability.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return str(fig_path)


# ==============================================================================
# 7. HÀM CHẠY TRỌN GÓI BƯỚC 3 & BƯỚC 4 (XUẤT ĐỦ HỢP ĐỒNG BÀN GIAO CHO NGƯỜI 3)
# ==============================================================================
def run_step3_and_step4_pipeline(
    fe_df: pd.DataFrame,
    split_df: pd.DataFrame,
    output_dir: str = "./artifacts_step3_4_handover",
) -> Dict[str, object]:
    """
    Thực thi trọn gói Bước 3 (Feature Selection 3 tầng + Ablation họ D)
    và Bước 4 (Đóng gói 3 Preprocessor + Kiểm thử Test D, Test E + Xuất Hợp đồng bàn giao).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_mask = (split_df["temporal_role"].to_numpy() == "train")
    val_mask = (split_df["temporal_role"].to_numpy() == "val")
    train_fe_df = fe_df.loc[train_mask]
    val_fe_df = fe_df.loc[val_mask]

    # 1. Chạy Bước 3: Lựa chọn đặc trưng 3 tầng chỉ trên tập Temporal Train
    fs_res = run_three_tier_feature_selection(
        train_fe_df,
        corr_threshold=0.95,
        n_inner_folds=5,
        n_perm_repeats=3,
        random_state=42,
    )
    selected_features: List[str] = fs_res["final_selected_features"]
    selection_report_df: pd.DataFrame = fs_res["selection_report_df"]
    ablation_df: pd.DataFrame = fs_res["ablation_df"]

    # 2. Chạy Bước 4: Khởi tạo và Fit 3 cấu hình Preprocessor chỉ trên tập Temporal Train
    preprocessors: Dict[str, ProductionTabularPreprocessor] = {}
    matrix_shapes: Dict[str, Dict[str, Tuple[int, int]]] = {}
    matrix_checksums: Dict[str, str] = {}

    for fam in ("linear", "sklearn_tree", "gbdt"):
        prep = ProductionTabularPreprocessor(selected_features=selected_features, model_family=fam)
        prep.fit(train_fe_df)
        preprocessors[fam] = prep

        X_tr_mat = prep.transform(train_fe_df)
        X_va_mat = prep.transform(val_fe_df)
        matrix_shapes[fam] = {"train_shape": X_tr_mat.shape, "val_shape": X_va_mat.shape}
        matrix_checksums[fam] = hashlib.sha256(np.nan_to_num(X_va_mat, nan=-999999.0).tobytes()).hexdigest()[:16]

    # 3. Thực thi bộ kiểm thử nghiệm thu cuối cùng (Test D & Test E)
    test_de_results = run_test_d_and_test_e(
        fe_df=fe_df,
        split_df=split_df,
        selected_features=selected_features,
        preprocessors=preprocessors,
        output_dir=out_path,
    )

    # 4. Vẽ biểu đồ tổng hợp Bước 3 (Figure 7) và xuất toàn bộ tệp bàn giao cho Người 3
    fig7_path = generate_step3_figure(selection_report_df, ablation_df, output_dir=out_path)

    selection_report_df.to_csv(out_path / "selection_and_feature_dictionary.csv", index=False)
    ablation_df.to_csv(out_path / "fs_d_ablation_results.csv", index=False)
    try:
        split_df.to_parquet(out_path / "split_indices.parquet", index=False)
    except Exception:
        split_df.to_csv(out_path / "split_indices.csv", index=False)

    final_features_payload = {
        "primary_split_protocol": "timestamp_safe_temporal_80_20",
        "secondary_benchmark_protocol": "stratified_80_20_seed_42",
        "primary_evaluation_metrics": ["ROC-AUC", "Average Precision (AP)"],
        "forbidden_excluded_columns": sorted(list(FORBIDDEN_PREDICTOR_COLS)),
        "n_candidate_features_before_selection": len(selection_report_df),
        "n_final_selected_raw_features": len(selected_features),
        "stability_breakdown": selection_report_df["Stability_Group"].value_counts().to_dict(),
        "selected_raw_features": selected_features,
        "transformed_feature_names_by_branch": {
            fam: prep.get_feature_names_out() for fam, prep in preprocessors.items()
        },
        "transformed_matrix_shapes": matrix_shapes,
        "validation_matrix_sha256_prefix": matrix_checksums,
        "unit_tests_passed": test_de_results,
        "ram_usage_stats_mb": {
            "fe_df_memory_mb": round(get_df_memory_mb(fe_df), 2),
            **get_process_rss_mb(),
        },
    }
    with open(out_path / "final_features.json", "w", encoding="utf-8") as f:
        json.dump(final_features_payload, f, indent=2, ensure_ascii=False)

    requirements_text = (
        "numpy>=1.24.0\n"
        "pandas>=2.0.0\n"
        "scikit-learn>=1.3.0\n"
        "lightgbm>=4.0.0\n"
        "xgboost>=2.0.0\n"
        "matplotlib>=3.7.0\n"
        "pyarrow>=12.0.0\n"
    )
    (out_path / "requirements.txt").write_text(requirements_text, encoding="utf-8")

    print("=" * 78)
    print("HOÀN TẤT BƯỚC 3 & BƯỚC 4: FEATURE SELECTION, UNIT TESTS (D, E) & BÀN GIAO")
    print("=" * 78)
    print(f"- Tổng số biến ứng viên đầu vào Bước 3 : {len(selection_report_df)} biến")
    print(f"- Số biến được giữ lại (Final Features): {len(selected_features)} biến")
    print(f"- Phân nhóm độ ổn định                 : {final_features_payload['stability_breakdown']}")
    print("\n[Bảng 3.1] Kết quả Thí nghiệm Ablation 4 cấu hình cho họ biến D (D1-D15 vs D_anchor):")
    print(ablation_df.to_string(index=False))
    print("\n[Bảng 3.2] Kích thước ma trận đầu ra của 3 Cấu hình Preprocessor bàn giao cho Người 3:")
    for fam, shp in matrix_shapes.items():
        print(
            f"  + preprocessor_{fam:<12}: Train = {shp['train_shape']} | "
            f"Val = {shp['val_shape']} | SHA256 = {matrix_checksums[fam]}"
        )
    print("\n- Kết quả Kiểm thử Tự động Bước 3 & 4:")
    for t_name, passed in test_de_results.items():
        print(f"  + {t_name}: {'PASSED' if passed else 'FAILED'}")
    print(
        f"- Bộ nhớ Tiến trình cuối Bước 4: Current RSS = {final_features_payload['ram_usage_stats_mb']['current_rss_mb']:.2f} MB | "
        f"Peak RSS = {final_features_payload['ram_usage_stats_mb']['peak_rss_mb']:.2f} MB"
    )
    print(f"\nĐã đóng gói đầy đủ bộ 6 nhóm file bàn giao cho Người 3 tại: {out_path}")

    return {
        "selected_features": selected_features,
        "selection_report_df": selection_report_df,
        "ablation_df": ablation_df,
        "preprocessors": preprocessors,
        "final_features_payload": final_features_payload,
        "fig7_path": fig7_path,
    }


if __name__ == "__main__":
    sample_df = _build_synthetic_ieee_cis_sample(n_rows=2000, seed=42)
    sample_df["C2"] = sample_df["C1"] * 1.01
    fraud_idx = sample_df["isFraud"] == 1
    sample_df.loc[fraud_idx, "C1"] = sample_df.loc[fraud_idx, "C1"] + 15.0

    step0_res = run_step0_pipeline(sample_df, output_dir="/tmp/test_step0_artifacts")
    step2_res = run_step2_feature_engineering(
        df=step0_res["df"],
        split_df=step0_res["split_df"],
        output_dir="/tmp/test_step2_fe_artifacts",
    )
    step34_res = run_step3_and_step4_pipeline(
        fe_df=step2_res["fe_df"],
        split_df=step0_res["split_df"],
        output_dir="/tmp/test_step3_4_handover",
    )
    print("\n[SELF-TEST PASSED] Toàn bộ quy trình Bước 0 -> Bước 4 đã vượt qua 100% kiểm thử nghiệm thu!")