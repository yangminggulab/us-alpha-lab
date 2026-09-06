from __future__ import annotations

import pandas as pd

from us_alpha_lab.a_share_l3 import (
    build_minute_features,
    build_order_lifecycle,
    cluster_share_report,
    coverage_for_frames,
    lifecycle_quality_report,
    minute_quality_report,
    quote_volume_reconciliation,
    run_static_force_clustering,
)


def test_build_order_lifecycle_uses_ex_order_id_and_cancel_events() -> None:
    orders = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000000,
                "order_id": 1,
                "ex_order_id": 101,
                "order_type": "0",
                "order_code": "B",
                "price": 33000.0,
                "volume": 1000.0,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000100,
                "order_id": 2,
                "ex_order_id": 202,
                "order_type": "0",
                "order_code": "S",
                "price": 33100.0,
                "volume": 600.0,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000200,
                "order_id": 101,
                "ex_order_id": 303,
                "order_type": "0",
                "order_code": "S",
                "price": 33200.0,
                "volume": 500.0,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93001000,
                "trade_id": 1,
                "trade_code": "0",
                "order_code": "0",
                "bs_flag": "B",
                "price": 33100.0,
                "volume": 600.0,
                "ask_order_id": 202,
                "bid_order_id": 101,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93002000,
                "trade_id": 2,
                "trade_code": "C",
                "order_code": "0",
                "bs_flag": "\x00",
                "price": 0.0,
                "volume": 400.0,
                "ask_order_id": 0,
                "bid_order_id": 101,
            },
        ]
    )

    lifecycle = build_order_lifecycle(orders, trades)

    by_id = lifecycle.set_index("ex_order_id")
    assert by_id.loc[101, "terminal_state"] == "PARTIAL_FILLED_CANCELED"
    assert by_id.loc[101, "filled_volume"] == 600.0
    assert by_id.loc[101, "canceled_volume"] == 400.0
    assert by_id.loc[101, "remaining_volume"] == 0.0
    assert by_id.loc[101, "aggressor_filled_volume"] == 600.0
    assert by_id.loc[202, "terminal_state"] == "FULLY_FILLED"
    assert by_id.loc[303, "terminal_state"] == "UNFILLED_OPEN"


def test_lifecycle_quality_report_flags_bad_order_id_join() -> None:
    orders = pd.DataFrame(
        [
            {
                "ex_order_id": 10,
                "order_code": "B",
                "time": 93000000,
                "price": 10000.0,
                "volume": 100.0,
            }
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "trade_code": "0",
                "bs_flag": "B",
                "time": 93001000,
                "price": 10000.0,
                "volume": 100.0,
                "ask_order_id": 20,
                "bid_order_id": 10,
            }
        ]
    )

    lifecycle = build_order_lifecycle(orders, trades)
    quality = lifecycle_quality_report(orders, trades, lifecycle)

    assert quality["ask_fill_hit_rate"] == 0.0
    assert quality["bid_fill_hit_rate"] == 1.0
    assert quality["negative_remaining_orders"] == 0


def test_quote_volume_reconciliation_uses_only_executed_trades() -> None:
    trades = pd.DataFrame(
        [
            {"trade_code": "0", "volume": 100.0},
            {"trade_code": "0", "volume": 50.0},
            {"trade_code": "C", "volume": 90.0},
        ]
    )
    quotes = pd.DataFrame(
        [
            {"cum_volume": 0.0, "volume": 0.0},
            {"cum_volume": 100.0, "volume": 100.0},
            {"cum_volume": 150.0, "volume": 50.0},
        ]
    )

    report = quote_volume_reconciliation(trades, quotes)

    assert report["trade_fill_volume"] == 150.0
    assert report["quote_final_cum_volume"] == 150.0
    assert report["quote_positive_tick_volume"] == 150.0
    assert report["trade_quote_volume_diff_abs"] == 0.0


def test_coverage_for_frames_separates_trade_stocks_without_orders() -> None:
    orders = pd.DataFrame(
        [
            {"wind_code": "000001.SZ", "ex_order_id": 10},
            {"wind_code": "000002.SZ", "ex_order_id": 20},
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "wind_code": "000001.SZ",
                "trade_code": "0",
                "ask_order_id": 10,
                "bid_order_id": 20,
            },
            {
                "wind_code": "600000.SH",
                "trade_code": "0",
                "ask_order_id": 30,
                "bid_order_id": 40,
            },
            {
                "wind_code": "000002.SZ",
                "trade_code": "C",
                "ask_order_id": 0,
                "bid_order_id": 20,
            },
        ]
    )

    report = coverage_for_frames("20170123", orders, trades)

    assert report["order_stocks"] == 2
    assert report["trade_stocks"] == 3
    assert report["trade_stocks_without_orders"] == 1
    assert report["sh_order_stocks"] == 0
    assert report["sh_trade_stocks"] == 1
    assert report["ask_fill_hit_rate"] == 0.5
    assert report["bid_fill_hit_rate"] == 0.0
    assert report["cancel_hit_rate"] == 1.0


def test_build_minute_features_combines_submit_and_event_observations() -> None:
    orders = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000000,
                "order_id": 1,
                "ex_order_id": 101,
                "order_type": "0",
                "order_code": "B",
                "price": 33000.0,
                "volume": 1000.0,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000100,
                "order_id": 2,
                "ex_order_id": 202,
                "order_type": "0",
                "order_code": "S",
                "price": 33100.0,
                "volume": 600.0,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000200,
                "order_id": 3,
                "ex_order_id": 303,
                "order_type": "0",
                "order_code": "S",
                "price": 33200.0,
                "volume": 500.0,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93001000,
                "trade_id": 1,
                "trade_code": "0",
                "order_code": "0",
                "bs_flag": "B",
                "price": 33100.0,
                "volume": 600.0,
                "ask_order_id": 202,
                "bid_order_id": 101,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93002000,
                "trade_id": 2,
                "trade_code": "C",
                "order_code": "0",
                "bs_flag": "\x00",
                "price": 0.0,
                "volume": 400.0,
                "ask_order_id": 0,
                "bid_order_id": 101,
            },
        ]
    )
    quotes = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000000,
                "price": 33000.0,
                "volume": 0.0,
                "cum_volume": 0.0,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93059000,
                "price": 33100.0,
                "volume": 600.0,
                "cum_volume": 600.0,
            },
        ]
    )

    lifecycle = build_order_lifecycle(orders, trades)
    minute = build_minute_features(lifecycle, trades, quotes=quotes)

    row = minute.iloc[0]
    assert row["minute"] == 570
    assert row["submit_order_count"] == 3
    assert row["submit_volume"] == 2100.0
    assert row["submit_buy_volume"] == 1000.0
    assert row["submit_sell_volume"] == 1100.0
    assert row["submit_filled_volume"] == 1200.0
    assert row["submit_canceled_volume"] == 400.0
    assert row["submit_remaining_volume"] == 500.0
    assert row["trade_volume"] == 600.0
    assert row["trade_active_buy_volume"] == 600.0
    assert row["event_cancel_buy_volume"] == 400.0
    assert row["quote_cum_volume"] == 600.0
    assert round(row["submit_buy_sell_imbalance"], 6) == round(-100 / 2100, 6)
    assert round(row["submit_order_hhi"], 6) == round(
        (1000 / 2100) ** 2 + (600 / 2100) ** 2 + (500 / 2100) ** 2,
        6,
    )


def test_build_minute_features_repairs_missing_aggressor_side_from_order_time() -> None:
    orders = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000000,
                "order_id": 1,
                "ex_order_id": 11,
                "order_type": "0",
                "order_code": "S",
                "price": 33100.0,
                "volume": 100.0,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000100,
                "order_id": 2,
                "ex_order_id": 22,
                "order_type": "0",
                "order_code": "B",
                "price": 33100.0,
                "volume": 100.0,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93001000,
                "trade_id": 1,
                "trade_code": "0",
                "order_code": "0",
                "bs_flag": "\x00",
                "price": 33100.0,
                "volume": 100.0,
                "ask_order_id": 11,
                "bid_order_id": 22,
            }
        ]
    )

    lifecycle = build_order_lifecycle(orders, trades)
    minute = build_minute_features(lifecycle, trades)

    row = minute.iloc[0]
    assert row["trade_missing_bs_flag_volume"] == 100.0
    assert row["trade_inferred_aggressor_volume"] == 100.0
    assert row["trade_unresolved_aggressor_volume"] == 0.0
    assert row["trade_active_buy_volume"] == 100.0
    assert row["trade_active_sell_volume"] == 0.0
    assert row["trade_active_imbalance"] == 1.0


def test_minute_quality_report_passes_clean_features_and_fails_bad_ratio() -> None:
    orders = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000000,
                "order_id": 1,
                "ex_order_id": 11,
                "order_type": "0",
                "order_code": "S",
                "price": 33100.0,
                "volume": 100.0,
            },
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93000100,
                "order_id": 2,
                "ex_order_id": 22,
                "order_type": "0",
                "order_code": "B",
                "price": 33100.0,
                "volume": 100.0,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "wind_code": "000725.SZ",
                "ex_code": "000725",
                "date": 20170123,
                "time": 93001000,
                "trade_id": 1,
                "trade_code": "0",
                "order_code": "0",
                "bs_flag": "B",
                "price": 33100.0,
                "volume": 100.0,
                "ask_order_id": 11,
                "bid_order_id": 22,
            }
        ]
    )
    lifecycle = build_order_lifecycle(orders, trades)
    minute = build_minute_features(lifecycle, trades)

    clean_report = minute_quality_report(lifecycle, trades, minute)
    assert clean_report["gate_status"] == "pass"
    assert clean_report["failure_count"] == 0

    broken = minute.copy()
    broken.loc[0, "submit_fill_ratio"] = 1.2
    broken_report = minute_quality_report(lifecycle, trades, broken)
    assert broken_report["gate_status"] == "fail"
    assert broken_report["ratio_out_of_bounds"] == 1


def test_run_static_force_clustering_selects_clear_two_cluster_shape() -> None:
    rows = []
    for index in range(6):
        rows.append(
            {
                "wind_code": "000001.SZ",
                "date": 20170123,
                "minute": 570 + index,
                "session": "continuous",
                "submit_buy_sell_imbalance": 0.8 + index * 0.01,
                "submit_order_hhi": 0.2,
                "submit_fill_ratio": 0.85,
                "submit_cancel_ratio": 0.05,
                "submit_open_ratio": 0.10,
                "submit_top_order_share": 0.3,
                "trade_active_imbalance": 0.9,
                "cancel_buy_sell_imbalance": -0.1,
                "submit_volume": 100_000 + index,
                "trade_volume": 80_000 + index,
                "event_cancel_volume": 5_000 + index,
                "submit_mean_lifetime_lower_bound_ms": 2_000 + index,
            }
        )
    for index in range(6):
        rows.append(
            {
                "wind_code": "000002.SZ",
                "date": 20170123,
                "minute": 570 + index,
                "session": "continuous",
                "submit_buy_sell_imbalance": -0.8 - index * 0.01,
                "submit_order_hhi": 0.7,
                "submit_fill_ratio": 0.25,
                "submit_cancel_ratio": 0.60,
                "submit_open_ratio": 0.15,
                "submit_top_order_share": 0.8,
                "trade_active_imbalance": -0.9,
                "cancel_buy_sell_imbalance": 0.6,
                "submit_volume": 20_000 + index,
                "trade_volume": 10_000 + index,
                "event_cancel_volume": 12_000 + index,
                "submit_mean_lifetime_lower_bound_ms": 15_000 + index,
            }
        )

    result = run_static_force_clustering(pd.DataFrame(rows), k_values=[2, 3], random_state=7)

    assert result.selected_k == 2
    assert set(result.labels["force_cluster"]) == {0, 1}
    assert "z_submit_buy_sell_imbalance" in result.labels.columns
    assert set(result.diagnostics["k"]) == {2, 3}
    assert len(result.profile) == 2
    assert result.profile["rows"].sum() == 12

    shares = cluster_share_report(result.labels)
    assert set(shares.columns) == {"date", "wind_code", "force_cluster", "rows", "share"}
    assert shares.groupby(["date", "wind_code"])["share"].sum().round(6).eq(1.0).all()
