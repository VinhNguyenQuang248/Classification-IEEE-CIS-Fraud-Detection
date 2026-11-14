"""
Bước 2 (Giai đoạn 2): Thiết kế Đặc trưng Chuẩn Point-in-Time (Point-in-Time Feature Engineering)
và Bộ Kiểm thử Tự động (Unit Tests A, B, C).

Đặc điểm kỹ thuật:
1. Nhóm biến Stateless (Tính độc lập trên từng dòng):
   - Tỷ lệ và số lượng khuyết theo 5 họ biến (C, D, M, V, và Identity có điều kiện khi has_identity == 1).
   - Chu kỳ thời gian tương đối (elapsed_hour, day_cycle, hour_sin/cos, day_sin/cos).
   - Biến neo mốc thời gian thử nghiệm D1_anchor..D15_anchor (elapsed_day - D_i) song song với D gốc.
   - Biến tiền tệ (TransactionAmt_log, Amt_cents, is_round_dollar).
   - Biến khớp email 3 trạng thái (1: trùng, 0: khác, -1: thiếu 1 hoặc cả 2) + nhà cung cấp + TLD.
   - Tách biệt hoàn toàn 3 trường thiết bị: device_brand (DeviceInfo), os_family (id_30), browser_family (id_31).
2. Nhóm biến Point-in-Time Expanding History (< t nghiêm ngặt cho card1):
   - Lịch sử nghiêm ngặt History(t) = {j : TransactionDT_j < t}, loại bỏ toàn bộ giao dịch cùng giây t.
   - GlobalMean(< t) cũng chỉ tính từ các giao dịch trước thời điểm t (không nhìn trước tương lai).
   - Xử lý rõ 3 trạng thái Cold-start: (1) có lịch sử, (2) thẻ mới dùng GlobalMean(< t), (3) card1 bị trống,
     cùng cờ no_global_history = 1 cho giây đầu tiên của lịch sử.
   - Hỗ trợ cả chế độ 'online' (cập nhật lịch sử quan sát trước t) và 'frozen' (đóng băng tại cuối Train).
3. Nhóm biến Frozen Batch Frequency Encoding (học từ Temporal Train cho các mã phân loại nhiều giá trị).
4. Bộ 3 Unit Tests Point-in-Time (Test A: Historical Recalculation, Test B: Future Perturbation,
   Test C: Timestamp Ties & Row-Order Invariance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from step0_setup_and_split import (
    _build_synthetic_ieee_cis_sample,
    get_ieee_cis_canonical_schema,
    run_step0_pipeline,
)


# ==============================================================================
# 1. CÁC HÀM CHUẨN HÓA CHUỖI THIẾT BỊ, HỆ ĐIỀU HÀNH, TRÌNH DUYỆT & EMAIL (STATELESS)
# ==============================================================================
def _clean_device_brand(series: pd.Series) -> pd.Series:
    """Chuẩn hóa DeviceInfo về nhóm hãng sản xuất / dòng thiết bị chính (tránh nổ hàng nghìn nhãn)."""
    s = series.astype("object").fillna("").astype(str).str.lower().str.strip()
    out = np.full(len(s), "Other", dtype=object)
    out[s == ""] = "Missing"
    out[s.str.contains("windows|trident|rv:|microsoft", regex=True)] = "Windows"
    out[s.str.contains("ios|iphone|ipad|ipod", regex=True)] = "iOS_Device"
    out[s.str.contains("macos|macintosh|mac os", regex=True)] = "MacOS"
    out[s.str.contains("sm-|samsung|gt-|sch-|sgh-", regex=True)] = "Samsung"
    out[s.str.contains("huawei|hi6210|ale-|cam-|pra-|bnd-", regex=True)] = "Huawei"
    out[s.str.contains("moto|xt1|xt2", regex=True)] = "Motorola"
    out[s.str.contains("lg-|lgms|vs9|ls9", regex=True)] = "LG"
    return pd.Series(out, index=series.index, dtype="category")


def _clean_os_family(series: pd.Series) -> pd.Series:
    """Khai thác hệ điều hành từ cột id_30 (tách biệt với DeviceInfo)."""
    s = series.astype("object").fillna("").astype(str).str.lower().str.strip()
    out = np.full(len(s), "Other", dtype=object)
    out[s == ""] = "Missing"
    out[s.str.contains("windows", regex=False)] = "Windows"
    out[s.str.contains("ios", regex=False)] = "iOS"
    out[s.str.contains("mac", regex=False)] = "Mac"
    out[s.str.contains("android", regex=False)] = "Android"
    out[s.str.contains("linux", regex=False)] = "Linux"
    return pd.Series(out, index=series.index, dtype="category")


def _clean_browser_family(series: pd.Series) -> pd.Series:
    """Khai thác trình duyệt từ cột id_31 (tách biệt với DeviceInfo và id_30)."""
    s = series.astype("object").fillna("").astype(str).str.lower().str.strip()
    out = np.full(len(s), "Other", dtype=object)
    out[s == ""] = "Missing"
    out[s.str.contains("chrome", regex=False)] = "Chrome"
    out[s.str.contains("safari", regex=False) & ~s.str.contains("chrome", regex=False)] = "Safari"
    out[s.str.contains("ie|edge|trident", regex=True)] = "IE_Edge"
    out[s.str.contains("firefox", regex=False)] = "Firefox"
    out[s.str.contains("samsung", regex=False)] = "Samsung_Browser"
    out[s.str.contains("android", regex=False)] = "Android_Browser"
    return pd.Series(out, index=series.index, dtype="category")


def _extract_email_provider_and_tld(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Tách riêng nhóm nhà cung cấp dịch vụ email và đuôi tên miền (TLD)."""
    s = series.astype("object").fillna("").astype(str).str.lower().str.strip()

    provider = np.full(len(s), "other", dtype=object)
    provider[s == ""] = "missing"
    provider[s.str.contains("gmail|google", regex=True)] = "gmail"
    provider[s.str.contains("yahoo|ymail|rocketmail", regex=True)] = "yahoo"
    provider[s.str.contains("hotmail|outlook|live|msn", regex=True)] = "microsoft"
    provider[s.str.contains("icloud|me.com|mac.com", regex=True)] = "apple"
    provider[s.str.contains("aol", regex=False)] = "aol"
    provider[s.str.contains("anonymous", regex=False)] = "anonymous"

    # Tách đuôi TLD (ví dụ .com, .mx, .es, .net)
    tld = s.str.extract(r"\.([a-z]+)$", expand=False).fillna("")
    tld_grouped = np.where(
        tld == "",
        "missing",
        np.where(tld.isin(["com", "mx", "es", "net", "de", "fr", "uk", "jp"]), tld, "other_tld"),
    )

    return (
        pd.Series(provider, index=series.index, dtype="category"),
        pd.Series(tld_grouped, index=series.index, dtype="category"),
    )


# ==============================================================================
# 2. BỘ TẠO ĐẶC TRƯNG STATELESS (KHÔNG PHỤ THUỘC THỐNG KÊ TẬP HỢP)
# ==============================================================================
def generate_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo toàn bộ các biến Stateless (chỉ phụ thuộc vào từng dòng giao dịch hiện tại,
    0% rủi ro rò rỉ giữa các dòng hoặc nhìn trước tương lai).
    """
    out = df.copy()

    # 1. Đảm bảo đã có has_identity và Amt_cents từ Bước 0
    if "has_identity" not in out.columns:
        id_cols_all = [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"]
        present_id = [c for c in id_cols_all if c in out.columns]
        out["has_identity"] = out[present_id].notna().any(axis=1).astype("int8")

    if "Amt_cents" not in out.columns:
        amt_f64 = out["TransactionAmt"].to_numpy(dtype="float64")
        out["Amt_cents"] = (np.round(amt_f64 * 100.0).astype("int64") % 100).astype("int8")

    # 2. Số lượng và tỷ lệ khuyết chuẩn hóa theo từng họ biến (C, D, M, V, và Identity có điều kiện)
    c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in out.columns]
    d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in out.columns]
    m_cols = [f"M{i}" for i in range(1, 10) if f"M{i}" in out.columns]
    v_cols = [f"V{i}" for i in range(1, 340) if f"V{i}" in out.columns]
    id_cols = [f"id_{i:02d}" for i in range(1, 39) if f"id_{i:02d}" in out.columns] + [
        c for c in ["DeviceType", "DeviceInfo"] if c in out.columns
    ]

    out["n_missing_C"] = out[c_cols].isna().sum(axis=1).astype("int8")
    out["missing_rate_C"] = (out["n_missing_C"] / max(len(c_cols), 1)).astype("float32")

    out["n_missing_D"] = out[d_cols].isna().sum(axis=1).astype("int8")
    out["missing_rate_D"] = (out["n_missing_D"] / max(len(d_cols), 1)).astype("float32")

    out["n_missing_M"] = out[m_cols].isna().sum(axis=1).astype("int8")
    out["missing_rate_M"] = (out["n_missing_M"] / max(len(m_cols), 1)).astype("float32")

    out["n_missing_V"] = out[v_cols].isna().sum(axis=1).astype("int16")
    out["missing_rate_V"] = (out["n_missing_V"] / max(len(v_cols), 1)).astype("float32")

    # Tách biệt n_missing_id và has_identity:
    # Khi has_identity == 1: đếm số trường thiếu bên trong bản ghi Identity (0..39).
    # Khi has_identity == 0: gán -1 để không mã hóa trùng lại has_identity.
    raw_id_miss = out[id_cols].isna().sum(axis=1).to_numpy(dtype="int16")
    has_id_arr = out["has_identity"].to_numpy()
    out["n_missing_id"] = np.where(has_id_arr == 1, raw_id_miss, -1).astype("int8")
    out["missing_rate_id"] = np.where(
        has_id_arr == 1,
        raw_id_miss / max(len(id_cols), 1),
        -1.0,
    ).astype("float32")

    # 3. Đặc trưng thời gian chu kỳ tương đối từ TransactionDT (timedelta tính bằng giây)
    dt_sec = out["TransactionDT"].to_numpy(dtype="int64")
    elapsed_day = (dt_sec // 86400).astype("int32")
    elapsed_hour = ((dt_sec // 3600) % 24).astype("int8")
    day_cycle = (elapsed_day % 7).astype("int8")

    out["elapsed_day"] = elapsed_day  # Biến trung gian, sẽ bị loại khỏi ma trận X ở Bước 3
    out["elapsed_hour"] = elapsed_hour
    out["day_cycle"] = day_cycle
    out["hour_sin"] = np.sin(2.0 * np.pi * elapsed_hour / 24.0).astype("float32")
    out["hour_cos"] = np.cos(2.0 * np.pi * elapsed_hour / 24.0).astype("float32")
    out["day_sin"] = np.sin(2.0 * np.pi * day_cycle / 7.0).astype("float32")
    out["day_cos"] = np.cos(2.0 * np.pi * day_cycle / 7.0).astype("float32")

    # 4. Biến thử nghiệm neo mốc thời gian tương đối cho họ D (D1_anchor .. D15_anchor)
    elapsed_day_f32 = elapsed_day.astype("float32")
    for d_col in d_cols:
        out[f"{d_col}_anchor"] = (elapsed_day_f32 - out[d_col].astype("float32")).astype("float32")

    # 5. Biến tiền tệ (TransactionAmt_log & is_round_dollar)
    amt_f32 = out["TransactionAmt"].astype("float32").clip(lower=0.0)
    out["TransactionAmt_log"] = np.log1p(amt_f32).astype("float32")
    out["is_round_dollar"] = (out["Amt_cents"] == 0).astype("int8")

    # 6. Biến Email 3 trạng thái (1: trùng, 0: khác, -1: thiếu 1 hoặc cả 2) + Provider + TLD
    if "P_emaildomain" in out.columns and "R_emaildomain" in out.columns:
        p_raw = out["P_emaildomain"].astype("object")
        r_raw = out["R_emaildomain"].astype("object")
        both_present = p_raw.notna() & r_raw.notna() & (p_raw.astype(str) != "") & (r_raw.astype(str) != "")
        same_dom = p_raw.astype(str).str.lower() == r_raw.astype(str).str.lower()
        out["email_domain_match"] = np.where(~both_present, -1, np.where(same_dom, 1, 0)).astype("int8")

        out["P_email_provider"], out["P_email_tld"] = _extract_email_provider_and_tld(out["P_emaildomain"])
        out["R_email_provider"], out["R_email_tld"] = _extract_email_provider_and_tld(out["R_emaildomain"])

    # 7. Tách riêng 3 biến Thiết bị / Hệ điều hành / Trình duyệt
    if "DeviceInfo" in out.columns:
        out["device_brand"] = _clean_device_brand(out["DeviceInfo"])
    if "id_30" in out.columns:
        out["os_family"] = _clean_os_family(out["id_30"])
    if "id_31" in out.columns:
        out["browser_family"] = _clean_browser_family(out["id_31"])

    return out


# ==============================================================================
# 3. POINT-IN-TIME EXPANDING HISTORY (< t NGHIÊM NGẶT) CHO CARD1
# ==============================================================================
class PointInTimeCardHistoryTransformer:
    """
    Tính toán các đặc trưng lịch sử nghiêm ngặt trước thời điểm t:
        History(t) = { j : TransactionDT_j < t }
    Đảm bảo:
    1. Loại bỏ toàn bộ các giao dịch có cùng TransactionDT == t (Timestamp Ties).
    2. GlobalMean(< t) cũng chỉ được tính từ các giao dịch có TransactionDT < t.
    3. Xử lý 3 trạng thái Cold-start:
       - card1 đã xuất hiện trước t (cnt_prior > 0): dùng Bayesian smoothed mean + has_card1_history = 1.
       - card1 mới xuất hiện lần đầu (cnt_prior == 0): dùng GlobalMean(< t) + has_card1_history = 0.
       - card1 bị thiếu (NaN): giữ NaN + card1_is_missing = 1.
       - Giao dịch ở giây đầu tiên (chưa có GlobalMean(< t)): gán NaN + no_global_history = 1.
    4. Bất biến hoàn toàn với thứ tự dòng (Row-Order Invariance).
    """

    def __init__(self, smoothing_m: float = 10.0, mode: str = "online") -> None:
        assert mode in ("online", "frozen"), "mode phải là 'online' hoặc 'frozen'"
        self.smoothing_m = float(smoothing_m)
        self.mode = mode
        self.train_history_df_: Optional[pd.DataFrame] = None
        self.frozen_card_table_: Optional[pd.DataFrame] = None
        self.frozen_global_mean_: float = np.nan

    @staticmethod
    def _compute_strict_prior_tables(base_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Từ tập dữ liệu lịch sử base_df (chứa TransactionDT, card1, TransactionAmt),
        tính bảng tích lũy nghiêm ngặt trước từng mốc thời gian t:
        - global_ts_table: [TransactionDT, global_cnt_prior, global_sum_prior, global_mean_prior]
        - card_ts_table  : [card1, TransactionDT, card1_cnt_prior, card1_sum_prior]
        """
        valid_amt = base_df["TransactionAmt"].notna() & np.isfinite(base_df["TransactionAmt"].to_numpy(dtype="float64"))
        clean_df = base_df.loc[valid_amt, ["TransactionDT", "card1", "TransactionAmt"]].copy()
        clean_df["TransactionDT"] = clean_df["TransactionDT"].astype("int64")
        clean_df["TransactionAmt"] = clean_df["TransactionAmt"].astype("float64")

        # 1. Lịch sử toàn cục (Global History < t) theo từng mốc TransactionDT duy nhất
        g_ts = (
            clean_df.groupby("TransactionDT", as_index=False)["TransactionAmt"]
            .agg(ts_cnt="count", ts_sum="sum")
            .sort_values("TransactionDT", kind="mergesort")
            .reset_index(drop=True)
        )
        g_ts["global_cnt_prior"] = g_ts["ts_cnt"].cumsum() - g_ts["ts_cnt"]
        g_ts["global_sum_prior"] = g_ts["ts_sum"].cumsum() - g_ts["ts_sum"]
        g_ts["global_mean_prior"] = np.where(
            g_ts["global_cnt_prior"] > 0,
            g_ts["global_sum_prior"] / np.maximum(g_ts["global_cnt_prior"], 1),
            np.nan,
        )

        # 2. Lịch sử theo từng (card1, TransactionDT) cho các dòng card1 không bị thiếu
        card_valid = clean_df[clean_df["card1"].notna()].copy()
        card_ts = (
            card_valid.groupby(["card1", "TransactionDT"], as_index=False)["TransactionAmt"]
            .agg(c_cnt="count", c_sum="sum")
            .sort_values(["card1", "TransactionDT"], kind="mergesort")
            .reset_index(drop=True)
        )
        card_ts["card1_cnt_prior"] = card_ts.groupby("card1")["c_cnt"].cumsum() - card_ts["c_cnt"]
        card_ts["card1_sum_prior"] = card_ts.groupby("card1")["c_sum"].cumsum() - card_ts["c_sum"]

        return (
            g_ts[["TransactionDT", "global_cnt_prior", "global_sum_prior", "global_mean_prior"]],
            card_ts[["card1", "TransactionDT", "card1_cnt_prior", "card1_sum_prior"]],
        )

    def fit(self, train_df: pd.DataFrame) -> "PointInTimeCardHistoryTransformer":
        """Lưu lịch sử giao dịch của tập Train để phục vụ chế độ 'online' hoặc 'frozen'."""
        cols = ["TransactionID", "TransactionDT", "card1", "TransactionAmt"]
        present_cols = [c for c in cols if c in train_df.columns]
        self.train_history_df_ = train_df[present_cols].copy()

        # Tính thống kê đóng băng ở cuối tập Train (phục vụ chế độ 'frozen')
        valid = self.train_history_df_["TransactionAmt"].notna()
        sub = self.train_history_df_.loc[valid]
        self.frozen_global_mean_ = float(sub["TransactionAmt"].astype("float64").mean()) if len(sub) > 0 else np.nan
        self.frozen_card_table_ = (
            sub[sub["card1"].notna()]
            .groupby("card1", as_index=False)["TransactionAmt"]
            .agg(frozen_cnt="count", frozen_sum="sum")
        )
        return self

    def transform(self, df: pd.DataFrame, is_train_split: bool = False) -> pd.DataFrame:
        """
        Biến đổi bảng df để gắn các cột lịch sử Point-in-Time của card1:
        - Nếu is_train_split=True hoặc mode='online': tính nghiêm ngặt từ các giao dịch có TransactionDT < t.
        - Nếu is_train_split=False và mode='frozen': tra cứu từ thống kê đã đóng băng ở cuối tập Train.
        """
        assert self.train_history_df_ is not None, "Phải gọi .fit(train_df) trước khi .transform()!"
        out = df.copy()

        if is_train_split or self.mode == "online":
            # Kết hợp lịch sử Train (nếu df chứa các dòng Validation sau Train) mà không nhân bản dòng
            if is_train_split:
                combined_hist = out[["TransactionDT", "card1", "TransactionAmt"]].copy()
            else:
                if "TransactionID" in out.columns and "TransactionID" in self.train_history_df_.columns:
                    extra_train = self.train_history_df_[
                        ~self.train_history_df_["TransactionID"].isin(out["TransactionID"])
                    ][["TransactionDT", "card1", "TransactionAmt"]]
                else:
                    min_dt_cur = out["TransactionDT"].min()
                    extra_train = self.train_history_df_[
                        self.train_history_df_["TransactionDT"] < min_dt_cur
                    ][["TransactionDT", "card1", "TransactionAmt"]]
                combined_hist = pd.concat(
                    [extra_train, out[["TransactionDT", "card1", "TransactionAmt"]]],
                    ignore_index=True,
                )

            g_ts, card_ts = self._compute_strict_prior_tables(combined_hist)

            # Ghép theo khóa (TransactionDT) và (card1, TransactionDT) để bảo toàn tuyệt đối thứ tự dòng của out
            lookup_df = pd.DataFrame(
                {
                    "TransactionDT": out["TransactionDT"].astype("int64").to_numpy(),
                    "card1": out["card1"].to_numpy(),
                },
                index=out.index,
            )
            merged = lookup_df.merge(g_ts, on="TransactionDT", how="left")
            merged = merged.merge(card_ts, on=["card1", "TransactionDT"], how="left")

            g_cnt_prior = merged["global_cnt_prior"].fillna(0).to_numpy(dtype="float64")
            g_mean_prior = merged["global_mean_prior"].to_numpy(dtype="float64")
            c_cnt_prior = merged["card1_cnt_prior"].to_numpy(dtype="float64")
            c_sum_prior = merged["card1_sum_prior"].to_numpy(dtype="float64")
        else:
            # Chế độ Frozen cho Validation/Test: dùng toàn bộ tập Train làm lịch sử cố định
            assert self.frozen_card_table_ is not None
            lookup_df = pd.DataFrame({"card1": out["card1"].to_numpy()}, index=out.index)
            merged = lookup_df.merge(self.frozen_card_table_, on="card1", how="left")
            g_cnt_prior = np.full(len(out), len(self.train_history_df_), dtype="float64")
            g_mean_prior = np.full(len(out), self.frozen_global_mean_, dtype="float64")
            c_cnt_prior = merged["frozen_cnt"].fillna(0.0).to_numpy(dtype="float64")
            c_sum_prior = merged["frozen_sum"].fillna(0.0).to_numpy(dtype="float64")

        # Áp dụng quy tắc 3 trạng thái Cold-start
        card_missing_mask = out["card1"].isna().to_numpy()
        no_global_mask = (g_cnt_prior <= 0) | (~np.isfinite(g_mean_prior))

        # Khi card1 hợp lệ nhưng chưa xuất hiện ở những giây có giao dịch của chính nó, c_cnt_prior = 0
        c_cnt_clean = np.where(card_missing_mask, np.nan, np.nan_to_num(c_cnt_prior, nan=0.0))
        c_sum_clean = np.where(card_missing_mask, np.nan, np.nan_to_num(c_sum_prior, nan=0.0))

        has_card_hist = (~card_missing_mask) & (c_cnt_clean > 0)

        # Công thức Bayesian Smoothing:
        # Mean_{card1, t} = (Sum_{card1, <t} + m * GlobalMean_{<t}) / (Count_{card1, <t} + m)
        m = self.smoothing_m
        smoothed_mean = np.where(
            card_missing_mask | no_global_mask,
            np.nan,
            (c_sum_clean + m * g_mean_prior) / (c_cnt_clean + m),
        )

        amt_arr = out["TransactionAmt"].to_numpy(dtype="float64")
        amt_ratio = np.where(
            np.isfinite(smoothed_mean) & (smoothed_mean > 0),
            amt_arr / smoothed_mean,
            np.nan,
        )

        out["card1_is_missing"] = card_missing_mask.astype("int8")
        out["no_global_history"] = no_global_mask.astype("int8")
        out["has_card1_history"] = has_card_hist.astype("int8")
        out["card1_cnt_prior"] = c_cnt_clean.astype("float32")
        out["card1_cnt_prior_log"] = np.where(
            np.isfinite(c_cnt_clean), np.log1p(np.maximum(c_cnt_clean, 0.0)), np.nan
        ).astype("float32")
        out["card1_mean_amt_prior"] = smoothed_mean.astype("float32")
        out["Amt_to_card1_mean_prior"] = amt_ratio.astype("float32")

        return out


# ==============================================================================
# 4. FROZEN BATCH FREQUENCY ENCODER CHO BIẾN PHÂN LOẠI NHIỀU GIÁ TRỊ
# ==============================================================================
class FrozenFrequencyEncoder:
    """
    Mã hóa tần suất xuất hiện (proportion trên tập Train) cho các biến phân loại nhiều giá trị.
    - Học bảng tần suất chỉ trên Temporal Train.
    - Giá trị chưa từng xuất hiện ở Validation/Test nhận 0.0; giá trị NaN nhận -1.0.
    """

    def __init__(self, cols: List[str]) -> None:
        self.cols = list(cols)
        self.freq_maps_: Dict[str, Dict[object, float]] = {}

    def fit(self, train_df: pd.DataFrame) -> "FrozenFrequencyEncoder":
        self.freq_maps_.clear()
        n_train = max(len(train_df), 1)
        for col in self.cols:
            if col not in train_df.columns:
                continue
            vc = train_df[col].dropna().value_counts()
            self.freq_maps_[col] = (vc / float(n_train)).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, fmap in self.freq_maps_.items():
            if col not in out.columns:
                continue
            s = out[col]
            mapped = s.map(fmap).astype("float64").fillna(0.0)
            mapped = np.where(s.isna(), -1.0, mapped)
            out[f"{col}_freq"] = mapped.astype("float32")
        return out


# ==============================================================================
# 5. BỘ 3 BÀI KIỂM THỬ POINT-IN-TIME ĐỘC LẬP (UNIT TESTS A, B, C)
# ==============================================================================
def _naive_reference_card1_point_in_time(
    df: pd.DataFrame, row_idx: int, smoothing_m: float = 10.0
) -> Dict[str, float]:
    """
    Hàm tham chiếu độc lập (vét cạn từng dòng trên lịch sử < t) dùng để kiểm định Test A.
    Không dùng lại bất kỳ hàm groupby/cumsum nào của PointInTimeCardHistoryTransformer.
    """
    row = df.iloc[row_idx]
    t_cur = int(row["TransactionDT"])
    c_cur = row["card1"]
    amt_cur = float(row["TransactionAmt"])

    # Lọc nghiêm ngặt các giao dịch có TransactionDT < t_cur
    hist = df[(df["TransactionDT"].astype("int64") < t_cur) & df["TransactionAmt"].notna()]
    g_cnt = len(hist)

    if pd.isna(c_cur):
        return {
            "card1_is_missing": 1,
            "no_global_history": int(g_cnt == 0),
            "has_card1_history": 0,
            "card1_cnt_prior": np.nan,
            "card1_mean_amt_prior": np.nan,
            "Amt_to_card1_mean_prior": np.nan,
        }

    if g_cnt == 0:
        return {
            "card1_is_missing": 0,
            "no_global_history": 1,
            "has_card1_history": 0,
            "card1_cnt_prior": 0.0,
            "card1_mean_amt_prior": np.nan,
            "Amt_to_card1_mean_prior": np.nan,
        }

    g_mean = float(hist["TransactionAmt"].astype("float64").mean())
    c_hist = hist[hist["card1"] == c_cur]
    c_cnt = float(len(c_hist))
    c_sum = float(c_hist["TransactionAmt"].astype("float64").sum()) if c_cnt > 0 else 0.0

    mean_prior = (c_sum + smoothing_m * g_mean) / (c_cnt + smoothing_m)
    ratio_prior = amt_cur / mean_prior if mean_prior > 0 else np.nan

    return {
        "card1_is_missing": 0,
        "no_global_history": 0,
        "has_card1_history": int(c_cnt > 0),
        "card1_cnt_prior": c_cnt,
        "card1_mean_amt_prior": mean_prior,
        "Amt_to_card1_mean_prior": ratio_prior,
    }


def run_point_in_time_unit_tests(df: pd.DataFrame, smoothing_m: float = 10.0) -> Dict[str, bool]:
    """
    Thực thi 3 bài kiểm thử Point-in-Time bắt buộc (Test A, Test B, Test C):
    - Test A (Historical Recalculation): Đối chiếu kết quả Pipeline với hàm vét cạn độc lập
      trên đủ 4 nhóm biên: (1) đầu tập dữ liệu (no_global_history), (2) các dòng trùng timestamp,
      (3) dòng card1 mới xuất hiện lần đầu & card1 bị thiếu, (4) dòng ngẫu nhiên.
    - Test B (Future Perturbation): Thay đổi ngẫu nhiên TransactionAmt và card1 của toàn bộ dòng
      có TransactionDT >= t_cut, xác nhận các đặc trưng của mọi dòng trước t_cut không đổi 100%.
    - Test C (Timestamp Ties & Row-Order Invariance): Xác nhận 3 giao dịch cùng timestamp + cùng card1
      nhận cùng giá trị lịch sử (không tự bao hàm lẫn nhau) và bất biến khi xáo trộn thứ tự dòng.
    """
    pit = PointInTimeCardHistoryTransformer(smoothing_m=smoothing_m, mode="online")
    pit.fit(df)
    transformed = pit.transform(df, is_train_split=True)

    # --------------------------------------------------------------------------
    # TEST A: HISTORICAL RECALCULATION (SO SÁNH VỚI HÀM VÉT CẠN ĐỘC LẬP)
    # --------------------------------------------------------------------------
    min_dt = df["TransactionDT"].min()
    first_dt_indices = np.flatnonzero(df["TransactionDT"].to_numpy() == min_dt)[:5].tolist()

    # Tìm các dòng trùng timestamp
    dt_counts = df["TransactionDT"].value_counts()
    tied_dts = dt_counts[dt_counts > 1].index.tolist()
    tie_indices = (
        np.flatnonzero(df["TransactionDT"].isin(tied_dts[:3])).tolist()[:10]
        if tied_dts
        else []
    )

    # Tìm các dòng card1 bị thiếu và các dòng card1 xuất hiện lần đầu
    na_card_indices = np.flatnonzero(df["card1"].isna().to_numpy())[:10].tolist()
    cold_card_indices = np.flatnonzero(
        (transformed["has_card1_history"].to_numpy() == 0)
        & (transformed["card1_is_missing"].to_numpy() == 0)
    )[:10].tolist()

    rng = np.random.default_rng(42)
    random_indices = rng.choice(len(df), size=min(40, len(df)), replace=False).tolist()
    test_a_indices = sorted(set(first_dt_indices + tie_indices + na_card_indices + cold_card_indices + random_indices))

    for idx in test_a_indices:
        ref = _naive_reference_card1_point_in_time(df, idx, smoothing_m=smoothing_m)
        row_t = transformed.iloc[idx]
        # So sánh chính xác cho các cột cờ nguyên
        for flag_col in ["card1_is_missing", "no_global_history", "has_card1_history"]:
            assert int(row_t[flag_col]) == int(ref[flag_col]), (
                f"Test A thất bại tại dòng {idx}, cột {flag_col}: {row_t[flag_col]} != {ref[flag_col]}"
            )
        # So sánh số thực với sai số float32 cho các cột tính toán
        for num_col in ["card1_cnt_prior", "card1_mean_amt_prior", "Amt_to_card1_mean_prior"]:
            assert np.allclose(
                np.float32(row_t[num_col]),
                np.float32(ref[num_col]),
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
            ), f"Test A thất bại tại dòng {idx}, cột {num_col}: {row_t[num_col]} != {ref[num_col]}"

    # --------------------------------------------------------------------------
    # TEST B: FUTURE PERTURBATION (BIẾN ĐỘNG TƯƠNG LAI KHÔNG ẢNH HƯỞNG QUÁ KHỨ)
    # --------------------------------------------------------------------------
    t_cut = float(df["TransactionDT"].quantile(0.55))
    df_perturbed = df.copy()
    future_mask = df_perturbed["TransactionDT"] >= t_cut
    df_perturbed.loc[future_mask, "TransactionAmt"] = df_perturbed.loc[future_mask, "TransactionAmt"] * 50.0 + 999.0
    df_perturbed.loc[future_mask, "card1"] = 999999.0

    pit_pert = PointInTimeCardHistoryTransformer(smoothing_m=smoothing_m, mode="online")
    pit_pert.fit(df_perturbed)
    trans_perturbed = pit_pert.transform(df_perturbed, is_train_split=True)

    past_mask = df["TransactionDT"] < t_cut
    pit_cols = [
        "card1_is_missing",
        "no_global_history",
        "has_card1_history",
        "card1_cnt_prior",
        "card1_cnt_prior_log",
        "card1_mean_amt_prior",
        "Amt_to_card1_mean_prior",
    ]
    for col in pit_cols:
        a_orig = transformed.loc[past_mask, col].to_numpy()
        a_pert = trans_perturbed.loc[past_mask, col].to_numpy()
        assert np.allclose(a_orig, a_pert, rtol=0.0, atol=0.0, equal_nan=True), (
            f"Test B (Future Perturbation) thất bại ở cột {col}!"
        )

    # --------------------------------------------------------------------------
    # TEST C: TIMESTAMP TIES & ROW-ORDER INVARIANCE (TRÙNG GIÂY & XÁO TRỘN DÒNG)
    # --------------------------------------------------------------------------
    # C.1: Kiểm tra bảng mẫu 3 giao dịch A, B, C cùng TransactionDT = 10000 và cùng card1 = 123
    tie_toy = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [5000, 10000, 10000, 10000],
            "card1": [123.0, 123.0, 123.0, 123.0],
            "TransactionAmt": [100.0, 200.0, 500.0, 900.0],
        }
    )
    pit_toy = PointInTimeCardHistoryTransformer(smoothing_m=10.0, mode="online").fit(tie_toy)
    res_toy = pit_toy.transform(tie_toy, is_train_split=True)
    # Ba dòng cuối (TransactionDT == 10000) phải nhận cùng cnt_prior = 1 và mean_prior = 100.0
    tied_rows = res_toy[res_toy["TransactionDT"] == 10000]
    assert np.all(tied_rows["card1_cnt_prior"].to_numpy() == 1.0), "Test C thất bại: Cùng timestamp bị tự bao hàm!"
    assert np.allclose(tied_rows["card1_mean_amt_prior"].to_numpy(), 100.0), (
        "Test C thất bại: Trung bình lịch sử của các dòng trùng timestamp không đồng nhất!"
    )

    # C.2: Kiểm tra bất biến khi xáo trộn ngẫu nhiên thứ tự dòng của toàn bộ df
    shuffled_df = df.sample(frac=1.0, random_state=999).reset_index(drop=True)
    pit_shuf = PointInTimeCardHistoryTransformer(smoothing_m=smoothing_m, mode="online").fit(shuffled_df)
    trans_shuf = pit_shuf.transform(shuffled_df, is_train_split=True)

    orig_by_id = transformed.sort_values("TransactionID").reset_index(drop=True)
    shuf_by_id = trans_shuf.sort_values("TransactionID").reset_index(drop=True)
    for col in pit_cols:
        assert np.allclose(
            orig_by_id[col].to_numpy(),
            shuf_by_id[col].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        ), f"Test C (Row-Order Invariance) thất bại ở cột {col}!"

    return {
        "Test_A_Historical_Recalculation": True,
        "Test_B_Future_Perturbation": True,
        "Test_C_Timestamp_Ties_and_Order_Invariance": True,
    }


# ==============================================================================
# 6. HÀM CHẠY TRỌN GÓI BƯỚC 2 (PIPELINE ENTRYPOINT)
# ==============================================================================
def run_step2_feature_engineering(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    output_dir: str = "./artifacts_step2_fe",
    history_mode: str = "online",
) -> Dict[str, object]:
    """
    Thực thi trọn gói Bước 2 (Feature Engineering & Point-in-Time Unit Tests A, B, C):
    1. Chạy kiểm thử tự động Test A, Test B, Test C trước tiên.
    2. Tạo toàn bộ biến Stateless (khuyết theo họ, thời gian chu kỳ, D_anchor, tiền tệ, email, thiết bị).
    3. Fit PointInTimeCardHistoryTransformer và FrozenFrequencyEncoder chỉ trên Temporal Train,
       sau đó biến đổi cho cả Temporal Train và Temporal Validation.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_mask = (split_df["temporal_role"].to_numpy() == "train")
    val_mask = (split_df["temporal_role"].to_numpy() == "val")

    # 1. Chạy bộ 3 Unit Tests Point-in-Time (A, B, C) trên tập Temporal Train
    unit_test_results = run_point_in_time_unit_tests(df.loc[train_mask].copy(), smoothing_m=10.0)

    # 2. Tạo các biến Stateless cho toàn bảng (vì Stateless chỉ tính trên từng dòng độc lập)
    fe_df = generate_stateless_features(df)

    # 3. Fit và Transform đặc trưng lịch sử Point-in-Time (< t) của card1
    pit_transformer = PointInTimeCardHistoryTransformer(smoothing_m=10.0, mode=history_mode)
    pit_transformer.fit(fe_df.loc[train_mask])

    fe_train = pit_transformer.transform(fe_df.loc[train_mask], is_train_split=True)
    fe_val = pit_transformer.transform(fe_df.loc[val_mask], is_train_split=False)
    fe_full = pd.concat([fe_train, fe_val], axis=0).loc[df.index]

    # 4. Fit FrozenFrequencyEncoder chỉ trên Temporal Train cho các biến nhiều giá trị
    high_card_cols = ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    freq_encoder = FrozenFrequencyEncoder(cols=high_card_cols)
    freq_encoder.fit(fe_full.loc[train_mask])
    fe_full = freq_encoder.transform(fe_full)

    # Tổng hợp danh sách các biến mới vừa được tạo ở Bước 2
    new_cols = [c for c in fe_full.columns if c not in df.columns]
    summary_rows = []
    for col in new_cols:
        s_tr = fe_full.loc[train_mask, col]
        summary_rows.append(
            {
                "New_Feature": col,
                "Dtype": str(s_tr.dtype),
                "Train_Missing_Rate_pct": round(float(s_tr.isna().mean() * 100.0), 2),
                "Unique_Values_Train": int(s_tr.nunique(dropna=True)),
            }
        )
    new_features_summary_df = pd.DataFrame(summary_rows)
    new_features_summary_df.to_csv(out_path / "fe_step2_new_features_summary.csv", index=False)

    print("=" * 78)
    print("HOÀN TẤT BƯỚC 2: POINT-IN-TIME FEATURE ENGINEERING & UNIT TESTS (A, B, C)")
    print("=" * 78)
    print("- Kết quả kiểm thử Point-in-Time Unit Tests:")
    for t_name, passed in unit_test_results.items():
        print(f"  + {t_name}: {'PASSED' if passed else 'FAILED'}")
    print(f"- Số lượng đặc trưng mới được tạo thêm ở Bước 2: {len(new_cols)} biến")
    print(f"- Tổng số cột hiện tại của bảng dữ liệu        : {fe_full.shape[1]} cột")
    print("\n[Bảng tóm tắt các nhóm biến mới tạo ở Bước 2 (20 biến tiêu biểu đầu tiên)]:")
    print(new_features_summary_df.head(20).to_string(index=False))

    return {
        "fe_df": fe_full,
        "new_feature_names": new_cols,
        "new_features_summary_df": new_features_summary_df,
        "unit_test_results": unit_test_results,
        "pit_transformer": pit_transformer,
        "freq_encoder": freq_encoder,
    }


if __name__ == "__main__":
    sample_df = _build_synthetic_ieee_cis_sample(n_rows=2000, seed=42)
    step0_res = run_step0_pipeline(sample_df, output_dir="/tmp/test_step0_artifacts")

    step2_res = run_step2_feature_engineering(
        df=step0_res["df"],
        split_df=step0_res["split_df"],
        output_dir="/tmp/test_step2_fe_artifacts",
        history_mode="online",
    )
    assert all(step2_res["unit_test_results"].values()), "Có Unit Test Point-in-Time không đạt!"
    print("\n[SELF-TEST PASSED] Bước 2 vượt qua toàn bộ Unit Tests A, B, C 100%!")