from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STREAM_FILES = {
    "orders": "逐笔委托.parquet",
    "trades": "逐笔成交.parquet",
    "quotes": "行情.parquet",
}

ORDER_COLUMNS = [
    "wind_code",
    "ex_code",
    "date",
    "time",
    "order_id",
    "ex_order_id",
    "order_type",
    "order_code",
    "price",
    "volume",
]
TRADE_COLUMNS = [
    "wind_code",
    "ex_code",
    "date",
    "time",
    "trade_id",
    "trade_code",
    "order_code",
    "bs_flag",
    "price",
    "volume",
    "ask_order_id",
    "bid_order_id",
]
QUOTE_COLUMNS = [
    "wind_code",
    "ex_code",
    "date",
    "time",
    "price",
    "volume",
    "amount",
    "cum_volume",
    "cum_amount",
]
DEFAULT_STATIC_CLUSTER_FEATURES = [
    "submit_buy_sell_imbalance",
    "submit_order_hhi",
    "submit_fill_ratio",
    "submit_cancel_ratio",
    "submit_open_ratio",
    "submit_top_order_share",
    "trade_active_imbalance",
    "cancel_buy_sell_imbalance",
    "submit_volume",
    "trade_volume",
    "event_cancel_volume",
    "submit_mean_lifetime_lower_bound_ms",
]
DEFAULT_STATIC_CLUSTER_LOG_FEATURES = {
    "submit_volume",
    "trade_volume",
    "event_cancel_volume",
    "submit_mean_lifetime_lower_bound_ms",
}
DEFAULT_HMM_STATE_FEATURES = DEFAULT_STATIC_CLUSTER_FEATURES


@dataclass(frozen=True)
class AShareL3StockDay:
    orders: pd.DataFrame
    trades: pd.DataFrame
    quotes: pd.DataFrame | None = None


@dataclass(frozen=True)
class StaticForceClusterResult:
    labels: pd.DataFrame
    diagnostics: pd.DataFrame
    profile: pd.DataFrame
    feature_columns: list[str]
    selected_k: int


@dataclass(frozen=True)
class HMMForceRegimeResult:
    labels: pd.DataFrame
    diagnostics: pd.DataFrame
    profile: pd.DataFrame
    transitions: pd.DataFrame
    feature_columns: list[str]
    selected_k: int


@dataclass(frozen=True)
class StateSignalScreenResult:
    summary: pd.DataFrame
    cell_level: pd.DataFrame
    spread: pd.DataFrame


def read_stock_day(
    raw_dir: Path | str,
    date: str | int,
    wind_code: str,
    *,
    columns_orders: list[str] | None = None,
    columns_trades: list[str] | None = None,
    columns_quotes: list[str] | None = None,
    include_quotes: bool = False,
) -> AShareL3StockDay:
    """Read one stock-day from the raw A-share L2/L3 parquet archive."""
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise ImportError("read_stock_day requires pyarrow to filter parquet files.") from exc

    date_text = str(date)
    day_dir = Path(raw_dir) / date_text
    if not day_dir.exists():
        raise FileNotFoundError(f"missing A-share L2/L3 day directory: {day_dir}")

    order_columns = columns_orders or ORDER_COLUMNS
    trade_columns = columns_trades or TRADE_COLUMNS
    quote_columns = columns_quotes or QUOTE_COLUMNS
    stock_filter = ds.field("wind_code") == wind_code
    orders = ds.dataset(day_dir / STREAM_FILES["orders"], format="parquet").to_table(
        filter=stock_filter,
        columns=order_columns,
    )
    trades = ds.dataset(day_dir / STREAM_FILES["trades"], format="parquet").to_table(
        filter=stock_filter,
        columns=trade_columns,
    )
    quotes = None
    if include_quotes:
        quotes = ds.dataset(day_dir / STREAM_FILES["quotes"], format="parquet").to_table(
            filter=stock_filter,
            columns=quote_columns,
        )
    return AShareL3StockDay(
        orders=orders.to_pandas(),
        trades=trades.to_pandas(),
        quotes=quotes.to_pandas() if quotes is not None else None,
    )


def available_l2_l3_days(raw_dir: Path | str) -> list[str]:
    root = Path(raw_dir)
    if not root.exists():
        return []
    days = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        if all((path / filename).exists() for filename in STREAM_FILES.values()):
            days.append(path.name)
    return sorted(days)


def normalize_side(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("\x00", "").strip().upper()
    return text if text else pd.NA


def add_time_features(frame: pd.DataFrame, *, time_col: str = "time") -> pd.DataFrame:
    result = frame.copy()
    time_values = pd.to_numeric(result[time_col], errors="coerce").astype("Int64")
    result["time_ms"] = hhmmssmmm_to_ms(time_values)
    result["minute"] = (result["time_ms"] // 60_000).astype("Int64")
    result["session"] = np.select(
        [
            result["time_ms"] < hhmmssmmm_scalar_to_ms(93000000),
            result["time_ms"] >= hhmmssmmm_scalar_to_ms(145700000),
        ],
        ["opening_auction", "closing_auction"],
        default="continuous",
    )
    return result


def hhmmssmmm_to_ms(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("Int64")
    hour = numeric // 10_000_000
    minute = (numeric // 100_000) % 100
    second = (numeric // 1_000) % 100
    millisecond = numeric % 1_000
    return (hour * 3_600_000 + minute * 60_000 + second * 1_000 + millisecond).astype("Int64")


def hhmmssmmm_scalar_to_ms(value: int) -> int:
    hour = value // 10_000_000
    minute = (value // 100_000) % 100
    second = (value // 1_000) % 100
    millisecond = value % 1_000
    return hour * 3_600_000 + minute * 60_000 + second * 1_000 + millisecond


def build_order_lifecycle(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    price_scale: float = 10_000.0,
    cancel_trade_code: str = "C",
) -> pd.DataFrame:
    """Build per-order lifecycle rows by joining trades to orders on ex_order_id."""
    _require_columns(orders, ["ex_order_id", "order_code", "time", "price", "volume"], "orders")
    _require_columns(
        trades,
        ["trade_code", "bs_flag", "time", "price", "volume", "ask_order_id", "bid_order_id"],
        "trades",
    )

    base = add_time_features(orders).copy()
    base["side"] = base["order_code"].map(normalize_side)
    base["price_raw"] = base["price"]
    base["price_scaled"] = base["price"] / price_scale
    base["submitted_volume"] = base["volume"].astype(float)
    base = base.drop(columns=["volume"])

    trades_with_time = add_time_features(trades).copy()
    trades_with_time["trade_code_clean"] = trades_with_time["trade_code"].map(normalize_side)
    trades_with_time["bs_flag_clean"] = trades_with_time["bs_flag"].map(normalize_side)
    is_cancel = trades_with_time["trade_code_clean"].eq(cancel_trade_code)

    fills = trades_with_time.loc[~is_cancel].copy()
    fill_refs = pd.concat(
        [
            _order_refs(fills, "ask_order_id", side="S", event_type="fill"),
            _order_refs(fills, "bid_order_id", side="B", event_type="fill"),
        ],
        ignore_index=True,
    )
    if not fill_refs.empty:
        fill_refs["is_aggressor"] = fill_refs["side"].eq(fill_refs["bs_flag_clean"])
    fill_agg = _aggregate_events(fill_refs, "filled")

    cancels = trades_with_time.loc[is_cancel].copy()
    cancel_refs = pd.concat(
        [
            _order_refs(cancels, "ask_order_id", side="S", event_type="cancel"),
            _order_refs(cancels, "bid_order_id", side="B", event_type="cancel"),
        ],
        ignore_index=True,
    )
    cancel_agg = _aggregate_events(cancel_refs, "canceled")

    lifecycle = base.merge(fill_agg, on="ex_order_id", how="left").merge(
        cancel_agg,
        on="ex_order_id",
        how="left",
    )
    lifecycle = _fill_lifecycle_nulls(lifecycle)
    lifecycle["accounted_volume"] = lifecycle["filled_volume"] + lifecycle["canceled_volume"]
    lifecycle["remaining_volume"] = lifecycle["submitted_volume"] - lifecycle["accounted_volume"]
    lifecycle["fill_ratio"] = _safe_divide(lifecycle["filled_volume"], lifecycle["submitted_volume"])
    lifecycle["cancel_ratio"] = _safe_divide(lifecycle["canceled_volume"], lifecycle["submitted_volume"])
    lifecycle["aggressor_fill_ratio"] = _safe_divide(
        lifecycle["aggressor_filled_volume"],
        lifecycle["filled_volume"],
    )

    last_event_time = lifecycle[["last_filled_time_ms", "last_canceled_time_ms"]].max(axis=1)
    first_event_time = lifecycle[["first_filled_time_ms", "first_canceled_time_ms"]].min(axis=1)
    lifecycle["first_event_time_ms"] = first_event_time
    lifecycle["last_event_time_ms"] = last_event_time
    lifecycle["is_terminal_observed"] = lifecycle["remaining_volume"].abs().le(1e-9)
    lifecycle["terminal_time_ms"] = np.where(
        lifecycle["is_terminal_observed"],
        lifecycle["last_event_time_ms"],
        np.nan,
    )
    lifecycle["lifetime_ms"] = lifecycle["terminal_time_ms"] - lifecycle["time_ms"].astype(float)
    lifecycle["observed_lifetime_lower_bound_ms"] = (
        lifecycle["last_event_time_ms"].fillna(lifecycle["time_ms"]).astype(float)
        - lifecycle["time_ms"].astype(float)
    )
    lifecycle["terminal_state"] = _terminal_state(lifecycle)
    return lifecycle


def lifecycle_quality_report(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    lifecycle: pd.DataFrame,
    *,
    quotes: pd.DataFrame | None = None,
    cancel_trade_code: str = "C",
) -> dict[str, float | int]:
    order_ids = set(pd.to_numeric(orders["ex_order_id"], errors="coerce").dropna().astype("uint64"))
    trade_code = trades["trade_code"].map(normalize_side)
    is_cancel = trade_code.eq(cancel_trade_code)
    fills = trades.loc[~is_cancel]
    cancels = trades.loc[is_cancel]

    ask_fill_hit = fills["ask_order_id"].isin(order_ids) & fills["ask_order_id"].ne(0)
    bid_fill_hit = fills["bid_order_id"].isin(order_ids) & fills["bid_order_id"].ne(0)
    cancel_ref = _nonzero_reference(cancels["ask_order_id"]).combine_first(
        _nonzero_reference(cancels["bid_order_id"])
    )
    cancel_hit = cancel_ref.isin(order_ids)

    report: dict[str, float | int] = {
        "orders": len(orders),
        "trades": len(trades),
        "fill_trades": len(fills),
        "cancel_events": len(cancels),
        "ask_fill_hit_rate": float(ask_fill_hit.mean()) if len(fills) else np.nan,
        "bid_fill_hit_rate": float(bid_fill_hit.mean()) if len(fills) else np.nan,
        "cancel_hit_rate": float(cancel_hit.mean()) if len(cancels) else np.nan,
        "negative_remaining_orders": int(lifecycle["remaining_volume"].lt(-1e-9).sum()),
        "terminal_observed_orders": int(lifecycle["is_terminal_observed"].sum()),
        "open_orders": int((~lifecycle["is_terminal_observed"]).sum()),
        "submitted_volume": float(lifecycle["submitted_volume"].sum()),
        "filled_volume": float(lifecycle["filled_volume"].sum()),
        "canceled_volume": float(lifecycle["canceled_volume"].sum()),
        "remaining_volume": float(lifecycle["remaining_volume"].sum()),
    }
    if quotes is not None:
        report.update(quote_volume_reconciliation(trades, quotes, cancel_trade_code=cancel_trade_code))
    return report


def quote_volume_reconciliation(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    cancel_trade_code: str = "C",
) -> dict[str, float | int]:
    _require_columns(trades, ["trade_code", "volume"], "trades")
    _require_columns(quotes, ["cum_volume", "volume"], "quotes")

    trade_code = trades["trade_code"].map(normalize_side)
    fills = trades.loc[~trade_code.eq(cancel_trade_code)]
    trade_fill_volume = float(pd.to_numeric(fills["volume"], errors="coerce").fillna(0.0).sum())
    quote_cum = pd.to_numeric(quotes["cum_volume"], errors="coerce").fillna(0.0)
    quote_tick_volume = pd.to_numeric(quotes["volume"], errors="coerce").fillna(0.0)
    quote_final_cum_volume = float(quote_cum.max()) if len(quote_cum) else 0.0
    quote_positive_tick_volume = float(quote_tick_volume.clip(lower=0.0).sum())
    quote_positive_cum_diff_volume = float(quote_cum.diff().fillna(quote_cum).clip(lower=0.0).sum())
    diff = trade_fill_volume - quote_final_cum_volume
    return {
        "quote_rows": len(quotes),
        "trade_fill_volume": trade_fill_volume,
        "quote_final_cum_volume": quote_final_cum_volume,
        "quote_positive_tick_volume": quote_positive_tick_volume,
        "quote_positive_cum_diff_volume": quote_positive_cum_diff_volume,
        "trade_quote_volume_diff": diff,
        "trade_quote_volume_diff_abs": abs(diff),
        "trade_quote_volume_diff_ratio": abs(diff) / trade_fill_volume if trade_fill_volume else np.nan,
    }


def build_minute_features(
    lifecycle: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    quotes: pd.DataFrame | None = None,
    price_scale: float = 10_000.0,
    cancel_trade_code: str = "C",
) -> pd.DataFrame:
    """Aggregate one stock-day lifecycle and trade events into minute observations."""
    _require_columns(
        lifecycle,
        [
            "wind_code",
            "date",
            "minute",
            "session",
            "side",
            "price_scaled",
            "submitted_volume",
            "filled_volume",
            "canceled_volume",
            "remaining_volume",
            "is_terminal_observed",
            "lifetime_ms",
            "observed_lifetime_lower_bound_ms",
            "terminal_state",
            "ex_order_id",
            "time_ms",
        ],
        "lifecycle",
    )
    _require_columns(
        trades,
        ["wind_code", "date", "time", "trade_code", "bs_flag", "price", "volume", "ask_order_id", "bid_order_id"],
        "trades",
    )

    keys = ["wind_code", "date", "minute"]
    submit = lifecycle.copy()
    submit["submitted_notional"] = submit["price_scaled"] * submit["submitted_volume"]
    submit_base = submit.groupby(keys, dropna=False).agg(
        session=("session", "first"),
        submit_order_count=("submitted_volume", "size"),
        submit_volume=("submitted_volume", "sum"),
        submit_notional=("submitted_notional", "sum"),
        submit_filled_volume=("filled_volume", "sum"),
        submit_canceled_volume=("canceled_volume", "sum"),
        submit_remaining_volume=("remaining_volume", "sum"),
        submit_terminal_observed_rate=("is_terminal_observed", "mean"),
        submit_mean_order_volume=("submitted_volume", "mean"),
        submit_max_order_volume=("submitted_volume", "max"),
        submit_mean_lifetime_ms=("lifetime_ms", "mean"),
        submit_mean_lifetime_lower_bound_ms=("observed_lifetime_lower_bound_ms", "mean"),
        submit_open_order_count=("terminal_state", lambda values: values.astype(str).str.endswith("_OPEN").sum()),
    )
    side_volume = _side_volume_pivot(submit, keys, "side", "submitted_volume", "submit")
    side_count = _side_count_pivot(submit, keys, "side", "submit")
    state_count = _state_count_pivot(submit, keys)
    hhi = _hhi_by_group(submit, keys, "submitted_volume").rename("submit_order_hhi")
    minute = (
        submit_base.join(side_volume, how="left")
        .join(side_count, how="left")
        .join(state_count, how="left")
        .join(hhi, how="left")
        .reset_index()
    )

    trade_events = _trade_minute_features(
        trades,
        lifecycle,
        price_scale=price_scale,
        cancel_trade_code=cancel_trade_code,
    )
    minute = minute.merge(trade_events, on=keys, how="outer")
    if quotes is not None:
        quote_features = _quote_minute_features(quotes, price_scale=price_scale)
        minute = minute.merge(quote_features, on=keys, how="outer")

    minute = minute.sort_values(keys).reset_index(drop=True)
    _ensure_columns(
        minute,
        {
            "submit_buy_volume": 0.0,
            "submit_sell_volume": 0.0,
            "submit_buy_order_count": 0,
            "submit_sell_order_count": 0,
            "submit_order_hhi": 0.0,
            "trade_count": 0,
            "trade_volume": 0.0,
            "trade_notional": 0.0,
            "trade_missing_bs_flag_volume": 0.0,
            "trade_inferred_aggressor_volume": 0.0,
            "trade_unresolved_aggressor_volume": 0.0,
            "trade_active_buy_volume": 0.0,
            "trade_active_sell_volume": 0.0,
            "event_cancel_count": 0,
            "event_cancel_volume": 0.0,
            "event_cancel_buy_volume": 0.0,
            "event_cancel_sell_volume": 0.0,
            "quote_ticks": 0,
            "quote_tick_volume": 0.0,
            "quote_cum_volume": 0.0,
        },
    )
    count_or_volume = [column for column in minute.columns if _is_zero_fill_feature(column)]
    minute[count_or_volume] = minute[count_or_volume].fillna(0.0)
    if "session" in minute:
        minute["session"] = minute["session"].fillna(_session_from_minute(minute["minute"]))

    minute["submit_buy_sell_imbalance"] = _safe_divide(
        minute.get("submit_buy_volume", 0.0) - minute.get("submit_sell_volume", 0.0),
        minute.get("submit_buy_volume", 0.0) + minute.get("submit_sell_volume", 0.0),
    )
    minute["submit_fill_ratio"] = _safe_divide(
        minute["submit_filled_volume"],
        minute["submit_volume"],
    )
    minute["submit_cancel_ratio"] = _safe_divide(
        minute["submit_canceled_volume"],
        minute["submit_volume"],
    )
    minute["submit_open_ratio"] = _safe_divide(
        minute["submit_remaining_volume"],
        minute["submit_volume"],
    )
    minute["submit_top_order_share"] = _safe_divide(
        minute["submit_max_order_volume"],
        minute["submit_volume"],
    )
    minute["trade_active_imbalance"] = _safe_divide(
        minute.get("trade_active_buy_volume", 0.0) - minute.get("trade_active_sell_volume", 0.0),
        minute.get("trade_active_buy_volume", 0.0) + minute.get("trade_active_sell_volume", 0.0),
    )
    minute["cancel_buy_sell_imbalance"] = _safe_divide(
        minute.get("event_cancel_buy_volume", 0.0) - minute.get("event_cancel_sell_volume", 0.0),
        minute.get("event_cancel_buy_volume", 0.0) + minute.get("event_cancel_sell_volume", 0.0),
    )
    return minute


def minute_quality_report(
    lifecycle: pd.DataFrame,
    trades: pd.DataFrame,
    minute_features: pd.DataFrame,
    *,
    quotes: pd.DataFrame | None = None,
    date: str | int | None = None,
    wind_code: str | None = None,
    tolerance: float = 1e-9,
    cancel_trade_code: str = "C",
) -> dict[str, float | int | str]:
    _require_columns(
        lifecycle,
        [
            "submitted_volume",
            "filled_volume",
            "canceled_volume",
            "remaining_volume",
            "is_terminal_observed",
        ],
        "lifecycle",
    )
    _require_columns(
        minute_features,
        [
            "submit_volume",
            "submit_filled_volume",
            "submit_canceled_volume",
            "submit_remaining_volume",
            "trade_volume",
            "event_cancel_volume",
            "submit_fill_ratio",
            "submit_cancel_ratio",
            "submit_open_ratio",
            "trade_active_imbalance",
            "cancel_buy_sell_imbalance",
            "trade_missing_bs_flag_volume",
            "trade_inferred_aggressor_volume",
            "trade_unresolved_aggressor_volume",
        ],
        "minute_features",
    )
    _require_columns(trades, ["trade_code", "volume"], "trades")

    trade_code = trades["trade_code"].map(normalize_side)
    trade_fill_volume = float(trades.loc[~trade_code.eq(cancel_trade_code), "volume"].sum())
    cancel_volume = float(trades.loc[trade_code.eq(cancel_trade_code), "volume"].sum())

    submit_volume_diff = float(lifecycle["submitted_volume"].sum() - minute_features["submit_volume"].sum())
    submit_filled_diff = float(
        lifecycle["filled_volume"].sum() - minute_features["submit_filled_volume"].sum()
    )
    submit_canceled_diff = float(
        lifecycle["canceled_volume"].sum() - minute_features["submit_canceled_volume"].sum()
    )
    submit_remaining_diff = float(
        lifecycle["remaining_volume"].sum() - minute_features["submit_remaining_volume"].sum()
    )
    trade_volume_diff = float(trade_fill_volume - minute_features["trade_volume"].sum())
    cancel_event_volume_diff = float(cancel_volume - minute_features["event_cancel_volume"].sum())

    quote_diff = np.nan
    if quotes is not None:
        quote_report = quote_volume_reconciliation(
            trades,
            quotes,
            cancel_trade_code=cancel_trade_code,
        )
        quote_diff = float(quote_report["trade_quote_volume_diff"])

    ratio_columns = ["submit_fill_ratio", "submit_cancel_ratio", "submit_open_ratio"]
    imbalance_columns = [
        "submit_buy_sell_imbalance",
        "trade_active_imbalance",
        "cancel_buy_sell_imbalance",
    ]
    ratio_out_of_bounds = _count_out_of_bounds(minute_features, ratio_columns, 0.0, 1.0, tolerance)
    imbalance_out_of_bounds = _count_out_of_bounds(
        minute_features,
        imbalance_columns,
        -1.0,
        1.0,
        tolerance,
    )
    critical_columns = [
        "submit_volume",
        "submit_fill_ratio",
        "submit_cancel_ratio",
        "submit_open_ratio",
        "trade_volume",
        "event_cancel_volume",
    ]
    critical_nan_count = int(minute_features[critical_columns].isna().sum().sum())
    unresolved_aggressor_volume = float(minute_features["trade_unresolved_aggressor_volume"].sum())
    missing_bs_flag_volume = float(minute_features["trade_missing_bs_flag_volume"].sum())
    inferred_aggressor_volume = float(minute_features["trade_inferred_aggressor_volume"].sum())
    max_abs_diff = max(
        abs(submit_volume_diff),
        abs(submit_filled_diff),
        abs(submit_canceled_diff),
        abs(submit_remaining_diff),
        abs(trade_volume_diff),
        abs(cancel_event_volume_diff),
        abs(quote_diff) if not pd.isna(quote_diff) else 0.0,
    )

    failure_count = int(
        lifecycle["remaining_volume"].lt(-tolerance).sum()
        + ratio_out_of_bounds
        + imbalance_out_of_bounds
        + critical_nan_count
        + (max_abs_diff > tolerance)
    )
    unresolved_ratio = (
        unresolved_aggressor_volume / missing_bs_flag_volume if missing_bs_flag_volume else 0.0
    )
    warning_count = int(unresolved_aggressor_volume > tolerance)
    status = "fail" if failure_count else "review" if warning_count else "pass"

    return {
        "date": str(date) if date is not None else _single_value(lifecycle, "date"),
        "wind_code": wind_code or _single_value(lifecycle, "wind_code"),
        "gate_status": status,
        "failure_count": failure_count,
        "warning_count": warning_count,
        "minute_rows": len(minute_features),
        "feature_columns": len(minute_features.columns),
        "terminal_observed_rate": float(lifecycle["is_terminal_observed"].mean()),
        "negative_remaining_orders": int(lifecycle["remaining_volume"].lt(-tolerance).sum()),
        "submit_volume_diff_abs": abs(submit_volume_diff),
        "submit_filled_diff_abs": abs(submit_filled_diff),
        "submit_canceled_diff_abs": abs(submit_canceled_diff),
        "submit_remaining_diff_abs": abs(submit_remaining_diff),
        "trade_volume_diff_abs": abs(trade_volume_diff),
        "cancel_event_volume_diff_abs": abs(cancel_event_volume_diff),
        "quote_trade_volume_diff_abs": abs(quote_diff) if not pd.isna(quote_diff) else np.nan,
        "ratio_out_of_bounds": ratio_out_of_bounds,
        "imbalance_out_of_bounds": imbalance_out_of_bounds,
        "critical_nan_count": critical_nan_count,
        "missing_bs_flag_volume": missing_bs_flag_volume,
        "inferred_aggressor_volume": inferred_aggressor_volume,
        "unresolved_aggressor_volume": unresolved_aggressor_volume,
        "unresolved_aggressor_ratio": unresolved_ratio,
        "opening_auction_minutes": int(minute_features["session"].eq("opening_auction").sum()),
        "continuous_minutes": int(minute_features["session"].eq("continuous").sum()),
        "closing_auction_minutes": int(minute_features["session"].eq("closing_auction").sum()),
        "max_submit_open_ratio": float(minute_features["submit_open_ratio"].max()),
        "max_submit_cancel_ratio": float(minute_features["submit_cancel_ratio"].max()),
        "max_submit_order_hhi": float(minute_features.get("submit_order_hhi", pd.Series([0.0])).max()),
    }


def build_quality_gate_report(
    raw_dir: Path | str,
    dates: list[str],
    wind_codes: list[str],
) -> pd.DataFrame:
    rows = []
    for date in dates:
        for wind_code in wind_codes:
            stock_day = read_stock_day(raw_dir, date, wind_code, include_quotes=True)
            lifecycle = build_order_lifecycle(stock_day.orders, stock_day.trades)
            minute_features = build_minute_features(
                lifecycle,
                stock_day.trades,
                quotes=stock_day.quotes,
            )
            rows.append(
                minute_quality_report(
                    lifecycle,
                    stock_day.trades,
                    minute_features,
                    quotes=stock_day.quotes,
                    date=date,
                    wind_code=wind_code,
                )
            )
    return pd.DataFrame(rows)


def run_static_force_clustering(
    minute_features: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    k_values: list[int] | None = None,
    sessions: list[str] | None = None,
    clip_quantile: float = 0.01,
    random_state: int = 0,
) -> StaticForceClusterResult:
    """Cluster minute-level order-flow shapes with clipped z-scored observations."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise ImportError("run_static_force_clustering requires scikit-learn.") from exc

    _require_columns(minute_features, ["wind_code", "date", "minute", "session"], "minute_features")
    selected_features = [
        column
        for column in (feature_columns or DEFAULT_STATIC_CLUSTER_FEATURES)
        if column in minute_features.columns
    ]
    if not selected_features:
        raise ValueError("no usable static cluster feature columns were found")

    data = minute_features.copy()
    if sessions is not None:
        data = data.loc[data["session"].astype(str).isin(sessions)].copy()
    if len(data) < 3:
        raise ValueError("need at least 3 minute rows for static force clustering")

    observation = _prepare_cluster_matrix(
        data,
        selected_features,
        clip_quantile=clip_quantile,
    )
    matrix = observation[[f"z_{column}" for column in selected_features]].to_numpy()
    allowed_k = [k for k in (k_values or [2, 3, 4, 5, 6]) if 1 < k < len(data)]
    if not allowed_k:
        raise ValueError("no valid k values for the available minute rows")

    diagnostics_rows = []
    fitted: dict[int, KMeans] = {}
    for k in allowed_k:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(matrix)
        fitted[k] = model
        diagnostics_rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(matrix, labels)),
            }
        )
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(
        ["silhouette", "k"],
        ascending=[False, True],
    )
    selected_k = int(diagnostics.iloc[0]["k"])
    selected_model = fitted[selected_k]
    labels = observation.copy()
    labels["force_cluster"] = selected_model.labels_
    distances = selected_model.transform(matrix)
    labels["force_cluster_distance"] = distances[np.arange(len(labels)), selected_model.labels_]

    profile = _cluster_profile(labels, selected_features)
    return StaticForceClusterResult(
        labels=labels,
        diagnostics=diagnostics,
        profile=profile,
        feature_columns=selected_features,
        selected_k=selected_k,
    )


def cluster_share_report(labels: pd.DataFrame) -> pd.DataFrame:
    _require_columns(labels, ["date", "wind_code", "force_cluster"], "labels")
    counts = (
        labels.groupby(["date", "wind_code", "force_cluster"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = counts.groupby(["date", "wind_code"], dropna=False)["rows"].transform("sum")
    counts["share"] = counts["rows"] / totals
    return counts.sort_values(["date", "wind_code", "force_cluster"]).reset_index(drop=True)


def run_hmm_force_regimes(
    minute_features: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    k_values: list[int] | None = None,
    sessions: list[str] | None = None,
    clip_quantile: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-4,
    transition_smoothing: float = 0.05,
    covariance_floor: float = 1e-3,
    random_state: int = 0,
) -> HMMForceRegimeResult:
    """Fit diagonal Gaussian HMM regimes on minute-level order-flow features."""
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise ImportError("run_hmm_force_regimes requires scikit-learn for initialization.") from exc

    _require_columns(minute_features, ["wind_code", "date", "minute", "session"], "minute_features")
    selected_features = [
        column
        for column in (feature_columns or DEFAULT_HMM_STATE_FEATURES)
        if column in minute_features.columns
    ]
    if not selected_features:
        raise ValueError("no usable HMM feature columns were found")

    data = minute_features.copy()
    if sessions is not None:
        data = data.loc[data["session"].astype(str).isin(sessions)].copy()
    data = data.sort_values(["date", "wind_code", "minute"]).reset_index(drop=True)
    if len(data) < 3:
        raise ValueError("need at least 3 minute rows for HMM regime fitting")

    observation = _prepare_cluster_matrix(
        data,
        selected_features,
        clip_quantile=clip_quantile,
    )
    matrix = observation[[f"z_{column}" for column in selected_features]].to_numpy(dtype=float)
    sequences = _sequence_slices(observation)
    allowed_k = [k for k in (k_values or [2, 3, 4, 5, 6]) if 1 < k < len(data)]
    if not allowed_k:
        raise ValueError("no valid k values for the available minute rows")

    diagnostics_rows = []
    fitted: dict[int, dict[str, np.ndarray | float | int | bool]] = {}
    for k in allowed_k:
        initial_labels = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(matrix)
        params = _initialize_hmm_parameters(
            matrix,
            initial_labels,
            sequences,
            k=k,
            transition_smoothing=transition_smoothing,
            covariance_floor=covariance_floor,
        )
        log_likelihood = -np.inf
        converged = False
        iteration = 0
        for iteration in range(1, max_iter + 1):
            log_emit = _hmm_log_emission(matrix, params["means"], params["variances"])
            stats = _hmm_expectation(
                log_emit,
                np.log(params["startprob"]),
                np.log(params["transmat"]),
                sequences,
            )
            new_log_likelihood = float(stats["log_likelihood"])
            if np.isfinite(log_likelihood) and abs(new_log_likelihood - log_likelihood) <= tol:
                log_likelihood = new_log_likelihood
                converged = True
                break
            log_likelihood = new_log_likelihood
            params = _hmm_maximization(
                matrix,
                stats,
                transition_smoothing=transition_smoothing,
                covariance_floor=covariance_floor,
            )

        param_count = (k - 1) + k * (k - 1) + 2 * k * matrix.shape[1]
        bic = -2.0 * log_likelihood + param_count * np.log(len(matrix))
        diagnostics_rows.append(
            {
                "k": k,
                "log_likelihood": log_likelihood,
                "bic": float(bic),
                "iterations": iteration,
                "converged": bool(converged),
            }
        )
        fitted[k] = {
            **params,
            "log_likelihood": log_likelihood,
            "bic": float(bic),
            "iterations": iteration,
            "converged": bool(converged),
        }

    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(["bic", "k"]).reset_index(drop=True)
    selected_k = int(diagnostics.iloc[0]["k"])
    selected = fitted[selected_k]
    log_emit = _hmm_log_emission(matrix, selected["means"], selected["variances"])
    stats = _hmm_expectation(
        log_emit,
        np.log(selected["startprob"]),
        np.log(selected["transmat"]),
        sequences,
    )
    path = _hmm_viterbi(
        log_emit,
        np.log(selected["startprob"]),
        np.log(selected["transmat"]),
        sequences,
    )
    labels = observation.copy()
    posterior = stats["gamma"]
    labels["hmm_state"] = path
    labels["hmm_state_posterior"] = posterior[np.arange(len(labels)), path]
    for state in range(selected_k):
        labels[f"hmm_state_prob_{state}"] = posterior[:, state]

    profile = _hmm_profile(labels, selected_features)
    transitions = _transition_frame(selected["transmat"])
    return HMMForceRegimeResult(
        labels=labels,
        diagnostics=diagnostics,
        profile=profile,
        transitions=transitions,
        feature_columns=selected_features,
        selected_k=selected_k,
    )


def add_minute_forward_returns(
    frame: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5),
    *,
    price_col: str = "quote_last_price_scaled",
) -> pd.DataFrame:
    """Add clock-minute forward returns aligned within each stock-day.

    A forward h-minute return at minute ``t`` is ``price[t + h] / price[t] - 1``
    and is only defined when a continuous-session row exists at ``t + h`` on the
    same stock-day. Rows at auction/closing minutes and rows whose target minute
    is absent (lunch break, halt, end of session) get ``NaN`` automatically, which
    keeps returns from silently spanning non-trading gaps.
    """
    _require_columns(frame, ["wind_code", "date", "minute", "session", price_col], "frame")
    key_columns = ["wind_code", "date"]
    if frame.duplicated(key_columns + ["minute"]).any():
        raise ValueError("add_minute_forward_returns requires one row per (wind_code, date, minute)")

    result = frame.copy()
    result["_key_minute"] = pd.to_numeric(result["minute"], errors="coerce").astype("float64")
    result["_price"] = pd.to_numeric(result[price_col], errors="coerce")
    result["_session"] = result["session"].astype(str)

    for horizon in sorted({int(value) for value in horizons}):
        target = result[key_columns + ["_key_minute", "_session", "_price"]].copy()
        target = target.rename(
            columns={
                "_key_minute": "_target_minute",
                "_session": "_target_session",
                "_price": "_target_price",
            }
        )
        lookup = result[key_columns + ["_key_minute"]].copy()
        lookup["_target_minute"] = lookup["_key_minute"] + float(horizon)
        joined = lookup.merge(target, on=key_columns + ["_target_minute"], how="left")

        base_price = result["_price"]
        forward = joined["_target_price"] / base_price - 1.0
        valid = (
            base_price.gt(0.0)
            & base_price.notna()
            & joined["_target_price"].notna()
            & joined["_target_session"].eq("continuous")
        )
        result[f"future_return_{horizon}m"] = forward.where(valid)

    return result.drop(columns=["_key_minute", "_price", "_session"])


def state_signal_screen(
    frame: pd.DataFrame,
    *,
    state_col: str = "hmm_state",
    horizons: Iterable[int] = (1, 3, 5),
) -> StateSignalScreenResult:
    """Describe per-state forward returns to screen whether states differentiate direction.

    ``summary`` reports pooled statistics per (state, horizon); ``cell_level`` reports
    per (date, wind_code, state, horizon) means for eyeballing cross-cell consistency;
    ``spread`` reports the pooled top-minus-bottom state mean gap per horizon.
    """
    _require_columns(frame, [state_col], "frame")
    forward_columns = sorted(
        (
            f"future_return_{int(value)}m"
            for value in horizons
            if f"future_return_{int(value)}m" in frame.columns
        ),
        key=lambda column: int(column.split("_")[2].rstrip("m")),
    )
    if not forward_columns:
        raise ValueError(
            "no future_return_{h}m columns found; call add_minute_forward_returns first"
        )

    summary_rows = []
    cell_rows = []
    for state, group in frame.groupby(state_col, sort=True):
        for column in forward_columns:
            horizon = int(column.split("_")[2].rstrip("m"))
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            count = len(values)
            if count == 0:
                summary_rows.append(_empty_state_summary(state_col, state, horizon))
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if count > 1 else np.nan
            se = float(std / np.sqrt(count)) if count > 1 and np.isfinite(std) else np.nan
            t_stat = float(mean / se) if se and np.isfinite(se) else np.nan
            summary_rows.append(
                {
                    state_col: state,
                    "horizon_min": horizon,
                    "n": count,
                    "mean_ret_bps": mean * 10_000.0,
                    "std_ret_bps": std * 10_000.0 if np.isfinite(std) else np.nan,
                    "se_bps": se * 10_000.0 if np.isfinite(se) else np.nan,
                    "t": t_stat,
                    "pos_rate": float((values > 0.0).mean()),
                }
            )
            for (date, wind_code), stock_day in group.groupby(["date", "wind_code"], sort=True):
                cell_values = pd.to_numeric(stock_day[column], errors="coerce").dropna()
                cell_rows.append(
                    {
                        "date": date,
                        "wind_code": wind_code,
                        state_col: state,
                        "horizon_min": horizon,
                        "n": len(cell_values),
                        "mean_ret_bps": float(cell_values.mean() * 10_000.0) if len(cell_values) else np.nan,
                    }
                )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["horizon_min", state_col],
        ascending=[True, True],
    )
    cell_level = pd.DataFrame(cell_rows).sort_values(
        ["date", "wind_code", "horizon_min", state_col],
        ascending=[True, True, True, True],
    )
    spread_rows = []
    for column in forward_columns:
        horizon = int(column.split("_")[2].rstrip("m"))
        state_rows = summary.loc[summary["horizon_min"].eq(horizon) & summary["n"].gt(0)]
        if state_rows.empty:
            continue
        best = state_rows.loc[state_rows["mean_ret_bps"].idxmax()]
        worst = state_rows.loc[state_rows["mean_ret_bps"].idxmin()]
        spread_rows.append(
            {
                "horizon_min": horizon,
                "top_state": best[state_col],
                "top_mean_ret_bps": float(best["mean_ret_bps"]),
                "top_n": int(best["n"]),
                "bottom_state": worst[state_col],
                "bottom_mean_ret_bps": float(worst["mean_ret_bps"]),
                "bottom_n": int(worst["n"]),
                "spread_bps": float(best["mean_ret_bps"] - worst["mean_ret_bps"]),
            }
        )
    spread = pd.DataFrame(spread_rows).sort_values("horizon_min")
    return StateSignalScreenResult(
        summary=summary,
        cell_level=cell_level,
        spread=spread,
    )


def _empty_state_summary(
    state_col: str,
    state: object,
    horizon: int,
    count: int,
) -> dict[str, float | int | object]:
    return {
        state_col: state,
        "horizon_min": horizon,
        "n": count,
        "mean_ret_bps": np.nan,
        "std_ret_bps": np.nan,
        "se_bps": np.nan,
        "t": np.nan,
        "pos_rate": np.nan,
    }


def _trade_minute_features(
    trades: pd.DataFrame,
    lifecycle: pd.DataFrame,
    *,
    price_scale: float,
    cancel_trade_code: str,
) -> pd.DataFrame:
    events = add_time_features(trades).copy()
    events["trade_code_clean"] = events["trade_code"].map(normalize_side)
    events["bs_flag_clean"] = events["bs_flag"].map(normalize_side)
    events["price_scaled"] = events["price"] / price_scale
    events["trade_notional"] = events["price_scaled"] * events["volume"]
    keys = ["wind_code", "date", "minute"]
    is_cancel = events["trade_code_clean"].eq(cancel_trade_code)

    fills = events.loc[~is_cancel].copy()
    fills = _add_fill_aggressor_side(fills, lifecycle)
    fill_base = fills.groupby(keys, dropna=False).agg(
        trade_count=("volume", "size"),
        trade_volume=("volume", "sum"),
        trade_notional=("trade_notional", "sum"),
        trade_missing_bs_flag_volume=(
            "volume",
            lambda values: values[fills.loc[values.index, "bs_flag_clean"].isna()].sum(),
        ),
        trade_inferred_aggressor_volume=(
            "volume",
            lambda values: values[fills.loc[values.index, "aggressor_side_source"].eq("order_time")].sum(),
        ),
        trade_unresolved_aggressor_volume=(
            "volume",
            lambda values: values[fills.loc[values.index, "aggressor_side"].isna()].sum(),
        ),
        trade_bs_flag_observed_rate=("bs_flag_observed", "mean"),
    )
    fill_base["trade_vwap_scaled"] = _safe_divide(
        fill_base["trade_notional"],
        fill_base["trade_volume"],
    )
    active_volume = _side_volume_pivot(fills, keys, "aggressor_side", "volume", "trade_active")

    cancels = events.loc[is_cancel].copy()
    cancel_base = cancels.groupby(keys, dropna=False).agg(
        event_cancel_count=("volume", "size"),
        event_cancel_volume=("volume", "sum"),
    )
    cancel_refs = pd.concat(
        [
            _order_refs(cancels, "ask_order_id", side="S", event_type="cancel"),
            _order_refs(cancels, "bid_order_id", side="B", event_type="cancel"),
        ],
        ignore_index=True,
    )
    cancel_side_volume = _side_volume_pivot(
        cancel_refs,
        keys,
        "side",
        "volume",
        "event_cancel",
    )

    frames = [fill_base, active_volume, cancel_base, cancel_side_volume]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=keys)
    result = pd.concat(frames, axis=1).reset_index()
    return result


def _prepare_cluster_matrix(
    minute_features: pd.DataFrame,
    feature_columns: list[str],
    *,
    clip_quantile: float,
) -> pd.DataFrame:
    if not 0.0 <= clip_quantile < 0.5:
        raise ValueError("clip_quantile must be in [0, 0.5)")

    key_columns = [
        column
        for column in ["wind_code", "date", "minute", "session", "quote_last_price_scaled"]
        if column in minute_features.columns
    ]
    result = minute_features[key_columns].copy()
    raw = minute_features[feature_columns].apply(pd.to_numeric, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    for column in feature_columns:
        raw_values = raw[column].copy()
        median = raw_values.median()
        if pd.isna(median):
            median = 0.0
        raw_values = raw_values.fillna(median)
        if clip_quantile > 0:
            lower = raw_values.quantile(clip_quantile)
            upper = raw_values.quantile(1.0 - clip_quantile)
            raw_values = raw_values.clip(lower=lower, upper=upper)
        transformed = raw_values.copy()
        if column in DEFAULT_STATIC_CLUSTER_LOG_FEATURES:
            transformed = np.log1p(transformed.clip(lower=0.0))
        result[column] = raw_values
        std = float(transformed.std(ddof=0))
        if std <= 0 or pd.isna(std):
            std = 1.0
        result[f"z_{column}"] = (transformed - float(transformed.mean())) / std
    return result


def _cluster_profile(labels: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for cluster, group in labels.groupby("force_cluster", sort=True):
        row: dict[str, float | int | str] = {
            "force_cluster": int(cluster),
            "rows": len(group),
            "share": len(group) / len(labels),
            "mean_distance": float(group["force_cluster_distance"].mean()),
            "dominant_session": str(group["session"].mode().iloc[0]) if "session" in group else "",
        }
        for column in feature_columns:
            row[f"mean_{column}"] = float(group[column].mean())
        rows.append(row)
    profile = pd.DataFrame(rows)
    return profile.sort_values(["rows", "force_cluster"], ascending=[False, True]).reset_index(drop=True)


def _sequence_slices(observation: pd.DataFrame) -> list[slice]:
    keys = ["date", "wind_code"]
    sequences = []
    start = 0
    for _, group in observation.groupby(keys, sort=False, dropna=False):
        stop = start + len(group)
        if stop > start:
            sequences.append(slice(start, stop))
        start = stop
    return sequences


def _initialize_hmm_parameters(
    matrix: np.ndarray,
    initial_labels: np.ndarray,
    sequences: list[slice],
    *,
    k: int,
    transition_smoothing: float,
    covariance_floor: float,
) -> dict[str, np.ndarray]:
    start_counts = np.full(k, transition_smoothing, dtype=float)
    transition_counts = np.full((k, k), transition_smoothing, dtype=float)
    for seq in sequences:
        labels = initial_labels[seq]
        start_counts[labels[0]] += 1.0
        if len(labels) > 1:
            np.add.at(transition_counts, (labels[:-1], labels[1:]), 1.0)

    means = np.zeros((k, matrix.shape[1]), dtype=float)
    variances = np.zeros((k, matrix.shape[1]), dtype=float)
    global_mean = matrix.mean(axis=0)
    global_var = matrix.var(axis=0) + covariance_floor
    for state in range(k):
        members = matrix[initial_labels == state]
        if len(members):
            means[state] = members.mean(axis=0)
            variances[state] = members.var(axis=0) + covariance_floor
        else:
            means[state] = global_mean
            variances[state] = global_var
    return {
        "startprob": _normalize_vector(start_counts),
        "transmat": _normalize_rows(transition_counts),
        "means": means,
        "variances": np.maximum(variances, covariance_floor),
    }


def _hmm_log_emission(matrix: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    safe_variances = np.maximum(variances, 1e-12)
    diff = matrix[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.log(2.0 * np.pi * safe_variances).sum(axis=1)[None, :]
        + ((diff**2) / safe_variances[None, :, :]).sum(axis=2)
    )


def _hmm_expectation(
    log_emit: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
    sequences: list[slice],
) -> dict[str, np.ndarray | float]:
    n_obs, k = log_emit.shape
    gamma = np.zeros((n_obs, k), dtype=float)
    xi_sum = np.zeros((k, k), dtype=float)
    start_gamma = np.zeros(k, dtype=float)
    log_likelihood = 0.0

    for seq in sequences:
        emissions = log_emit[seq]
        alpha = np.zeros_like(emissions)
        beta = np.zeros_like(emissions)
        alpha[0] = log_start + emissions[0]
        for t in range(1, len(emissions)):
            alpha[t] = emissions[t] + _logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
        sequence_log_likelihood = float(_logsumexp(alpha[-1], axis=0))
        log_likelihood += sequence_log_likelihood

        for t in range(len(emissions) - 2, -1, -1):
            beta[t] = _logsumexp(log_trans + emissions[t + 1][None, :] + beta[t + 1][None, :], axis=1)

        seq_gamma = np.exp(alpha + beta - sequence_log_likelihood)
        seq_gamma = np.clip(seq_gamma, 0.0, 1.0)
        seq_gamma = _normalize_rows(seq_gamma)
        gamma[seq] = seq_gamma
        start_gamma += seq_gamma[0]
        for t in range(len(emissions) - 1):
            xi_log = (
                alpha[t][:, None]
                + log_trans
                + emissions[t + 1][None, :]
                + beta[t + 1][None, :]
                - sequence_log_likelihood
            )
            xi_sum += np.exp(xi_log)

    return {
        "gamma": gamma,
        "xi_sum": xi_sum,
        "start_gamma": start_gamma,
        "log_likelihood": log_likelihood,
    }


def _hmm_maximization(
    matrix: np.ndarray,
    stats: dict[str, np.ndarray | float],
    *,
    transition_smoothing: float,
    covariance_floor: float,
) -> dict[str, np.ndarray]:
    gamma = stats["gamma"]
    xi_sum = stats["xi_sum"]
    start_gamma = stats["start_gamma"]
    weights = gamma.sum(axis=0).clip(min=1e-12)
    means = (gamma.T @ matrix) / weights[:, None]
    centered = matrix[:, None, :] - means[None, :, :]
    variances = (gamma[:, :, None] * centered**2).sum(axis=0) / weights[:, None]
    return {
        "startprob": _normalize_vector(start_gamma + transition_smoothing),
        "transmat": _normalize_rows(xi_sum + transition_smoothing),
        "means": means,
        "variances": np.maximum(variances, covariance_floor),
    }


def _hmm_viterbi(
    log_emit: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
    sequences: list[slice],
) -> np.ndarray:
    path = np.zeros(len(log_emit), dtype=int)
    for seq in sequences:
        emissions = log_emit[seq]
        delta = np.zeros_like(emissions)
        back = np.zeros_like(emissions, dtype=int)
        delta[0] = log_start + emissions[0]
        for t in range(1, len(emissions)):
            scores = delta[t - 1][:, None] + log_trans
            back[t] = np.argmax(scores, axis=0)
            delta[t] = emissions[t] + scores[back[t], np.arange(log_emit.shape[1])]
        seq_path = np.zeros(len(emissions), dtype=int)
        seq_path[-1] = int(np.argmax(delta[-1]))
        for t in range(len(emissions) - 2, -1, -1):
            seq_path[t] = back[t + 1, seq_path[t + 1]]
        path[seq] = seq_path
    return path


def _hmm_profile(labels: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    run_lengths = _state_run_lengths(labels)
    rows = []
    for state, group in labels.groupby("hmm_state", sort=True):
        row: dict[str, float | int | str] = {
            "hmm_state": int(state),
            "rows": len(group),
            "share": len(group) / len(labels),
            "mean_posterior": float(group["hmm_state_posterior"].mean()),
            "mean_duration_minutes": float(run_lengths.get(int(state), pd.Series([1.0])).mean()),
        }
        for column in feature_columns:
            row[f"mean_{column}"] = float(group[column].mean())
        rows.append(row)
    profile = pd.DataFrame(rows)
    return profile.sort_values(["rows", "hmm_state"], ascending=[False, True]).reset_index(drop=True)


def _state_run_lengths(labels: pd.DataFrame) -> dict[int, pd.Series]:
    lengths: dict[int, list[int]] = {}
    for _, group in labels.groupby(["date", "wind_code"], sort=False, dropna=False):
        states = group["hmm_state"].to_numpy()
        if len(states) == 0:
            continue
        start = 0
        for index in range(1, len(states) + 1):
            if index == len(states) or states[index] != states[start]:
                state = int(states[start])
                lengths.setdefault(state, []).append(index - start)
                start = index
    return {state: pd.Series(values, dtype=float) for state, values in lengths.items()}


def _transition_frame(transmat: np.ndarray) -> pd.DataFrame:
    rows = []
    for from_state in range(transmat.shape[0]):
        for to_state in range(transmat.shape[1]):
            rows.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "transition_prob": float(transmat[from_state, to_state]),
                }
            )
    return pd.DataFrame(rows)


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(len(values), 1.0 / len(values))
    return values / total


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sums, out=np.full_like(values, 1.0 / values.shape[1]), where=row_sums > 0)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_value).sum(axis=axis, keepdims=True)
    result = max_value + np.log(stable)
    return np.squeeze(result, axis=axis)


def _add_fill_aggressor_side(fills: pd.DataFrame, lifecycle: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        result = fills.copy()
        result["bs_flag_observed"] = pd.Series(dtype=bool)
        result["aggressor_side"] = pd.Series(dtype=object)
        result["aggressor_side_source"] = pd.Series(dtype=object)
        return result

    order_times = lifecycle[["ex_order_id", "time_ms"]].drop_duplicates("ex_order_id").copy()
    order_times["ex_order_id"] = pd.to_numeric(order_times["ex_order_id"], errors="coerce").astype("uint64")
    result = fills.merge(
        order_times.rename(columns={"ex_order_id": "ask_order_id", "time_ms": "ask_order_time_ms"}),
        on="ask_order_id",
        how="left",
    ).merge(
        order_times.rename(columns={"ex_order_id": "bid_order_id", "time_ms": "bid_order_time_ms"}),
        on="bid_order_id",
        how="left",
    )
    observed = result["bs_flag_clean"].isin(["B", "S"])
    inferred = pd.Series(pd.NA, index=result.index, dtype="object")
    ask_time = pd.to_numeric(result["ask_order_time_ms"], errors="coerce")
    bid_time = pd.to_numeric(result["bid_order_time_ms"], errors="coerce")
    inferred = inferred.mask(bid_time.gt(ask_time), "B")
    inferred = inferred.mask(ask_time.gt(bid_time), "S")
    result["bs_flag_observed"] = observed
    result["aggressor_side"] = result["bs_flag_clean"].where(observed, inferred)
    result["aggressor_side_source"] = np.select(
        [observed, result["aggressor_side"].notna()],
        ["bs_flag", "order_time"],
        default="unknown",
    )
    return result


def _quote_minute_features(quotes: pd.DataFrame, *, price_scale: float) -> pd.DataFrame:
    _require_columns(quotes, ["wind_code", "date", "time", "price", "volume", "cum_volume"], "quotes")
    frame = add_time_features(quotes).copy()
    frame["quote_price_scaled"] = frame["price"] / price_scale
    keys = ["wind_code", "date", "minute"]
    return (
        frame.groupby(keys, dropna=False)
        .agg(
            quote_ticks=("volume", "size"),
            quote_tick_volume=("volume", "sum"),
            quote_cum_volume=("cum_volume", "max"),
            quote_last_price_scaled=("quote_price_scaled", "last"),
        )
        .reset_index()
    )


def _side_volume_pivot(
    frame: pd.DataFrame,
    keys: list[str],
    side_column: str,
    value_column: str,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[f"{prefix}_buy_volume", f"{prefix}_sell_volume"])
    pivot = frame.pivot_table(
        index=keys,
        columns=side_column,
        values=value_column,
        aggfunc="sum",
        fill_value=0.0,
    )
    result = pd.DataFrame(index=pivot.index)
    result[f"{prefix}_buy_volume"] = pivot.get("B", 0.0)
    result[f"{prefix}_sell_volume"] = pivot.get("S", 0.0)
    return result


def _side_count_pivot(
    frame: pd.DataFrame,
    keys: list[str],
    side_column: str,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[f"{prefix}_buy_order_count", f"{prefix}_sell_order_count"])
    pivot = frame.pivot_table(
        index=keys,
        columns=side_column,
        values="submitted_volume",
        aggfunc="size",
        fill_value=0,
    )
    result = pd.DataFrame(index=pivot.index)
    result[f"{prefix}_buy_order_count"] = pivot.get("B", 0)
    result[f"{prefix}_sell_order_count"] = pivot.get("S", 0)
    return result


def _state_count_pivot(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot_table(
        index=keys,
        columns="terminal_state",
        values="submitted_volume",
        aggfunc="size",
        fill_value=0,
    )
    pivot.columns = [f"state_{str(column).lower()}_count" for column in pivot.columns]
    return pivot


def _hhi_by_group(frame: pd.DataFrame, keys: list[str], value_column: str) -> pd.Series:
    total = frame.groupby(keys, dropna=False)[value_column].transform("sum")
    shares = _safe_divide(frame[value_column], total)
    return shares.pow(2).groupby([frame[column] for column in keys], dropna=False).sum()


def _is_zero_fill_feature(column: str) -> bool:
    endings = ("_count", "_volume", "_notional", "_hhi")
    return column.endswith(endings) or column.startswith("state_")


def _session_from_minute(minute: pd.Series) -> pd.Series:
    minute_numeric = pd.to_numeric(minute, errors="coerce")
    return pd.Series(
        np.select(
            [minute_numeric < 570, minute_numeric >= 897],
            ["opening_auction", "closing_auction"],
            default="continuous",
        ),
        index=minute.index,
    )


def _count_out_of_bounds(
    frame: pd.DataFrame,
    columns: list[str],
    lower: float,
    upper: float,
    tolerance: float,
) -> int:
    total = 0
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        total += int((values.lt(lower - tolerance) | values.gt(upper + tolerance)).sum())
    return total


def _ensure_columns(frame: pd.DataFrame, defaults: dict[str, float | int | str]) -> None:
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default


def _single_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame or frame.empty:
        return ""
    values = frame[column].dropna().unique()
    return str(values[0]) if len(values) else ""


def coverage_for_frames(
    date: str | int,
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    cancel_trade_code: str = "C",
) -> dict[str, float | int | str]:
    _require_columns(orders, ["wind_code", "ex_order_id"], "orders")
    _require_columns(trades, ["wind_code", "trade_code", "ask_order_id", "bid_order_id"], "trades")

    order_stocks = set(orders["wind_code"].dropna().astype(str))
    trade_stocks = set(trades["wind_code"].dropna().astype(str))
    order_pairs = _order_pair_index(orders)
    trade_code = trades["trade_code"].map(normalize_side)
    fills = trades.loc[~trade_code.eq(cancel_trade_code)]
    cancels = trades.loc[trade_code.eq(cancel_trade_code)]

    ask_fill_hit = _pair_hit(fills, "ask_order_id", order_pairs)
    bid_fill_hit = _pair_hit(fills, "bid_order_id", order_pairs)
    cancel_hit = _pair_hit(cancels, "ask_order_id", order_pairs) | _pair_hit(
        cancels,
        "bid_order_id",
        order_pairs,
    )

    return {
        "date": str(date),
        "order_rows": len(orders),
        "trade_rows": len(trades),
        "order_stocks": len(order_stocks),
        "trade_stocks": len(trade_stocks),
        "trade_stocks_with_orders": len(order_stocks & trade_stocks),
        "trade_stocks_without_orders": len(trade_stocks - order_stocks),
        "sz_order_stocks": sum(code.endswith(".SZ") for code in order_stocks),
        "sh_order_stocks": sum(code.endswith(".SH") for code in order_stocks),
        "sz_trade_stocks": sum(code.endswith(".SZ") for code in trade_stocks),
        "sh_trade_stocks": sum(code.endswith(".SH") for code in trade_stocks),
        "fill_trades": len(fills),
        "cancel_events": len(cancels),
        "ask_fill_hit_rate": float(ask_fill_hit.mean()) if len(fills) else np.nan,
        "bid_fill_hit_rate": float(bid_fill_hit.mean()) if len(fills) else np.nan,
        "cancel_hit_rate": float(cancel_hit.mean()) if len(cancels) else np.nan,
    }


def build_l3_coverage_report(raw_dir: Path | str, dates: list[str] | None = None) -> pd.DataFrame:
    selected_days = dates or available_l2_l3_days(raw_dir)
    rows = [coverage_for_day(raw_dir, date) for date in selected_days]
    return pd.DataFrame(rows)


def coverage_for_day(raw_dir: Path | str, date: str | int) -> dict[str, float | int | str]:
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise ImportError("coverage_for_day requires pyarrow to read parquet files.") from exc

    day_dir = Path(raw_dir) / str(date)
    orders_table = ds.dataset(day_dir / STREAM_FILES["orders"], format="parquet").to_table(
        columns=["wind_code", "ex_order_id"],
    )
    trade_stocks_table = ds.dataset(day_dir / STREAM_FILES["trades"], format="parquet").to_table(
        columns=["wind_code"],
    )

    order_stocks = set(pc.unique(orders_table["wind_code"]).to_pylist())
    trade_stocks = set(pc.unique(trade_stocks_table["wind_code"]).to_pylist())
    order_stocks.discard(None)
    trade_stocks.discard(None)

    if order_stocks:
        trades_table = ds.dataset(day_dir / STREAM_FILES["trades"], format="parquet").to_table(
            filter=ds.field("wind_code").isin(sorted(order_stocks)),
            columns=["wind_code", "trade_code", "ask_order_id", "bid_order_id"],
        )
        trades_for_orders = trades_table.to_pandas()
    else:
        trades_for_orders = pd.DataFrame(
            columns=["wind_code", "trade_code", "ask_order_id", "bid_order_id"]
        )
    orders = orders_table.to_pandas()

    order_pairs = _order_pair_index(orders)
    trade_code = trades_for_orders["trade_code"].map(normalize_side)
    fills = trades_for_orders.loc[~trade_code.eq("C")]
    cancels = trades_for_orders.loc[trade_code.eq("C")]

    ask_fill_hit = _pair_hit(fills, "ask_order_id", order_pairs)
    bid_fill_hit = _pair_hit(fills, "bid_order_id", order_pairs)
    cancel_hit = _pair_hit(cancels, "ask_order_id", order_pairs) | _pair_hit(
        cancels,
        "bid_order_id",
        order_pairs,
    )

    return {
        "date": str(date),
        "order_rows": len(orders),
        "trade_rows": trade_stocks_table.num_rows,
        "trade_rows_with_orders": len(trades_for_orders),
        "order_stocks": len(order_stocks),
        "trade_stocks": len(trade_stocks),
        "trade_stocks_with_orders": len(order_stocks & trade_stocks),
        "trade_stocks_without_orders": len(trade_stocks - order_stocks),
        "sz_order_stocks": sum(str(code).endswith(".SZ") for code in order_stocks),
        "sh_order_stocks": sum(str(code).endswith(".SH") for code in order_stocks),
        "sz_trade_stocks": sum(str(code).endswith(".SZ") for code in trade_stocks),
        "sh_trade_stocks": sum(str(code).endswith(".SH") for code in trade_stocks),
        "fill_trades_with_orders": len(fills),
        "cancel_events_with_orders": len(cancels),
        "ask_fill_hit_rate": float(ask_fill_hit.mean()) if len(fills) else np.nan,
        "bid_fill_hit_rate": float(bid_fill_hit.mean()) if len(fills) else np.nan,
        "cancel_hit_rate": float(cancel_hit.mean()) if len(cancels) else np.nan,
    }


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _order_refs(frame: pd.DataFrame, id_column: str, *, side: str, event_type: str) -> pd.DataFrame:
    key_columns = [column for column in ["wind_code", "date", "minute"] if column in frame.columns]
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *key_columns,
                "ex_order_id",
                "side",
                "event_type",
                "time_ms",
                "price",
                "volume",
                "bs_flag_clean",
            ]
        )
    refs = frame.loc[
        frame[id_column].ne(0),
        [*key_columns, "time_ms", "price", "volume", id_column, "bs_flag_clean"],
    ].copy()
    refs = refs.rename(columns={id_column: "ex_order_id"})
    refs["side"] = side
    refs["event_type"] = event_type
    refs["ex_order_id"] = pd.to_numeric(refs["ex_order_id"], errors="coerce").astype("uint64")
    return refs


def _aggregate_events(events: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "ex_order_id",
                f"{prefix}_volume",
                f"{prefix}_events",
                f"first_{prefix}_time_ms",
                f"last_{prefix}_time_ms",
            ]
        )

    grouped = events.groupby("ex_order_id", sort=False)
    result = grouped.agg(
        **{
            f"{prefix}_volume": ("volume", "sum"),
            f"{prefix}_events": ("volume", "size"),
            f"first_{prefix}_time_ms": ("time_ms", "min"),
            f"last_{prefix}_time_ms": ("time_ms", "max"),
        }
    ).reset_index()
    if prefix == "filled":
        aggressor = (
            events.loc[events["is_aggressor"]]
            .groupby("ex_order_id")["volume"]
            .sum()
            .rename("aggressor_filled_volume")
            .reset_index()
        )
        passive = (
            events.loc[~events["is_aggressor"]]
            .groupby("ex_order_id")["volume"]
            .sum()
            .rename("passive_filled_volume")
            .reset_index()
        )
        result = result.merge(aggressor, on="ex_order_id", how="left").merge(
            passive,
            on="ex_order_id",
            how="left",
        )
    return result


def _fill_lifecycle_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    zero_columns = [
        "filled_volume",
        "filled_events",
        "aggressor_filled_volume",
        "passive_filled_volume",
        "canceled_volume",
        "canceled_events",
    ]
    time_columns = [
        "first_filled_time_ms",
        "last_filled_time_ms",
        "first_canceled_time_ms",
        "last_canceled_time_ms",
    ]
    for column in zero_columns:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = result[column].fillna(0.0)
    for column in time_columns:
        if column not in result.columns:
            result[column] = np.nan
    return result


def _terminal_state(frame: pd.DataFrame) -> pd.Series:
    remaining = frame["remaining_volume"]
    filled = frame["filled_volume"]
    canceled = frame["canceled_volume"]
    states = np.select(
        [
            remaining.lt(-1e-9),
            remaining.abs().le(1e-9) & filled.gt(0) & canceled.eq(0),
            remaining.abs().le(1e-9) & filled.eq(0) & canceled.gt(0),
            remaining.abs().le(1e-9) & filled.gt(0) & canceled.gt(0),
            remaining.gt(1e-9) & (filled.gt(0) | canceled.gt(0)),
        ],
        [
            "OVER_ACCOUNTED",
            "FULLY_FILLED",
            "FULLY_CANCELED",
            "PARTIAL_FILLED_CANCELED",
            "PARTIAL_OPEN",
        ],
        default="UNFILLED_OPEN",
    )
    return pd.Series(states, index=frame.index)


def _safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    if not isinstance(numerator, pd.Series):
        index = denominator.index if isinstance(denominator, pd.Series) else None
        numerator = pd.Series(numerator, index=index)
    if not isinstance(denominator, pd.Series):
        denominator = pd.Series(denominator, index=numerator.index)
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def _nonzero_reference(values: pd.Series) -> pd.Series:
    refs = pd.to_numeric(values, errors="coerce")
    return refs.where(refs.ne(0))


def _order_pair_index(orders: pd.DataFrame) -> pd.MultiIndex:
    refs = pd.DataFrame(
        {
            "wind_code": orders["wind_code"].astype(str),
            "ex_order_id": pd.to_numeric(orders["ex_order_id"], errors="coerce"),
        }
    ).dropna()
    refs = refs.loc[refs["ex_order_id"].ne(0)].copy()
    refs["ex_order_id"] = refs["ex_order_id"].astype("uint64")
    return pd.MultiIndex.from_frame(refs[["wind_code", "ex_order_id"]])


def _pair_hit(frame: pd.DataFrame, id_column: str, order_pairs: pd.MultiIndex) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    refs = pd.DataFrame(
        {
            "wind_code": frame["wind_code"].astype(str),
            "ex_order_id": pd.to_numeric(frame[id_column], errors="coerce"),
        },
        index=frame.index,
    )
    valid = refs["ex_order_id"].notna() & refs["ex_order_id"].ne(0)
    refs["ex_order_id"] = refs["ex_order_id"].fillna(0).astype("uint64")
    pairs = pd.MultiIndex.from_frame(refs[["wind_code", "ex_order_id"]])
    return pd.Series(pairs.isin(order_pairs) & valid.to_numpy(), index=frame.index)
