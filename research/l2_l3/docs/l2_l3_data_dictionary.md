# A-share L2/L3 Data Dictionary

> Current calibration sample: `20170123 / 000725.SZ`.
> Raw location: `../../data/raw/a_share_l2_hf/<YYYYMMDD>/`.

## Confirmed Design Rules

| Topic | Confirmed rule | Evidence / note |
| --- | --- | --- |
| Order-trade join key | Join `orders.ex_order_id` to `trades.ask_order_id` and `trades.bid_order_id` | Across 11 local days, SZ stocks with orders have 100% ask/bid fill hit rates and 100% cancel hit rates. |
| Do not join on `order_id` | `orders.order_id` is only local row/order sequence and can numerically collide with `ex_order_id` | Tests include a deliberate collision to keep this locked. |
| Exchange scope | Order stream currently covers SZ names only; SH names can still appear in trades/quotes | Lifecycle research is SZ-only unless another data source adds SH orders. |
| Cancel events | Explicit cancels are in `trades.trade_code == 'C'`, not in `orders.order_type` | Cancel rows have price 0, one nonzero side id, null `bs_flag`, and 100% hit `orders.ex_order_id` in the sample. |
| Price scale | Raw price appears scaled by 10,000 | `32800` corresponds to `3.28`; lifecycle output keeps `price_raw` and adds `price_scaled`. |
| Volume unit | Raw volume is shares | Trade amount / raw price / volume is about `1 / 10000`, consistent with price scale rather than volume lots. |
| `ex_code` | Stock code without exchange suffix, not participant/seat id | On `20170123`, every stock has exactly one `ex_code`, e.g. `000725.SZ -> 000725`. |
| Time format | Integer `HHMMSSmmm` with the leading zero dropped | Example: `91500010` means `09:15:00.010`; parsed to `time_ms`. |

## 逐笔委托.parquet

| Column | Meaning | Current interpretation |
| --- | --- | --- |
| `wind_code` | Wind security code | Includes suffix, e.g. `000725.SZ`. |
| `ex_code` | Exchange security code | Six-digit stock code without suffix; not a broker/seat dimension. |
| `date` | Trading date | `YYYYMMDD` integer. |
| `time` | Event time | `HHMMSSmmm` integer with dropped leading zero. |
| `order_id` | Local sequence id | Not suitable for joining with trades. |
| `ex_order_id` | Exchange order id | Primary key used by trade/cancel references. |
| `order_type` | Order type code | Mostly `0`; `1` and `U` exist but are not terminal-state labels. In the sample, `order_type='1'` orders can fill. |
| `order_code` | Order side | `B` buy, `S` sell. |
| `price` | Submitted price | Raw integer-like price scaled by 10,000; zero can appear on special order types. |
| `volume` | Submitted volume | Shares. |

## 逐笔成交.parquet

| Column | Meaning | Current interpretation |
| --- | --- | --- |
| `wind_code` | Wind security code | Includes suffix. |
| `ex_code` | Exchange security code | Six-digit stock code without suffix. |
| `date` | Trading date | `YYYYMMDD` integer. |
| `time` | Event time | Same `HHMMSSmmm` convention. |
| `trade_id` | Exchange trade/event id | Event sequence id. |
| `trade_code` | Trade event type | `0` = executed trade; `C` = cancellation event. |
| `order_code` | Trade-side code | In the sample this is `0`; use `bs_flag` and resting order side instead. |
| `bs_flag` | Aggressor side for executed trades | `B` active buy, `S` active sell; cancel rows are null/`\x00`. |
| `price` | Execution price | Scaled by 10,000; cancel rows have 0. |
| `volume` | Executed or canceled volume | Shares. |
| `ask_order_id` | Sell-side referenced `ex_order_id` | For fills, nonzero and should hit an order. For sell-side cancels, this is the canceled order id. |
| `bid_order_id` | Buy-side referenced `ex_order_id` | For fills, nonzero and should hit an order. For buy-side cancels, this is the canceled order id. |

## 行情.parquet

| Column group | Meaning | Current interpretation |
| --- | --- | --- |
| `price`, `high`, `low`, `open`, `prev_close` | Last/summary prices | Same 10,000 price scale. |
| `volume`, `amount`, `cum_volume`, `cum_amount` | Tick and cumulative activity | Volume in shares; amount in currency-scaled notional. |
| `ask_px1..10`, `bid_px1..10` | Ten-level book prices | Same 10,000 price scale. |
| `ask_vol1..10`, `bid_vol1..10` | Ten-level book volumes | Shares. |
| `wavg_ask_px`, `wavg_bid_px` | Weighted book prices | Same price scale. |
| `tot_ask_vol`, `tot_bid_vol` | Aggregate depth | Shares. |

## Lifecycle Output

Command:

```bash
alpha-lab a-share-l3-lifecycle --date 20170123 --wind-code 000725.SZ
```

Default output:

`reports/l2_l3/lifecycle_<date>_<ticker>.parquet`

Important columns:

| Column | Meaning |
| --- | --- |
| `submitted_volume` | Original order volume. |
| `filled_volume` | Sum of executed fills referencing this `ex_order_id`. |
| `canceled_volume` | Sum of `trade_code='C'` events referencing this `ex_order_id`. |
| `remaining_volume` | `submitted_volume - filled_volume - canceled_volume`; should never be negative beyond tolerance. |
| `aggressor_filled_volume` | Fill volume where order side equals `bs_flag`, i.e. the order acted as the aggressor. |
| `passive_filled_volume` | Fill volume where the opposite side was aggressor. |
| `terminal_state` | `FULLY_FILLED`, `FULLY_CANCELED`, `PARTIAL_FILLED_CANCELED`, `PARTIAL_OPEN`, `UNFILLED_OPEN`, or `OVER_ACCOUNTED`. |
| `lifetime_ms` | Submit time to final observed terminal event for fully accounted orders. |
| `observed_lifetime_lower_bound_ms` | Submit time to last observed event, or 0 for untouched open orders. |

## Minute Feature Output

Command:

```bash
alpha-lab a-share-l3-minute-features --date 20170123 --wind-code 000725.SZ
```

Default output:

`reports/l2_l3/minute_features_<date>_<ticker>.parquet`

The minute table intentionally separates two clocks:

| Prefix | Clock | Meaning |
| --- | --- | --- |
| `submit_*` | Order submission minute | What intentions were posted during the minute, and how those orders eventually ended. |
| `trade_*` | Executed trade minute | What actually traded during the minute, including active buy/sell flow. |
| `event_cancel_*` | Cancel event minute | What was explicitly canceled during the minute. |
| `quote_*` | Quote/tick minute | Market data totals and last observed quote price for the minute. |

Core feature groups:

| Feature | Meaning |
| --- | --- |
| `submit_buy_sell_imbalance` | `(buy submitted volume - sell submitted volume) / submitted volume`. |
| `submit_order_hhi` | HHI concentration of order sizes submitted during the minute. |
| `submit_fill_ratio` | Submitted orders' eventual filled volume divided by submitted volume. |
| `submit_cancel_ratio` | Submitted orders' eventual canceled volume divided by submitted volume. |
| `submit_open_ratio` | Submitted orders' end-of-day remaining volume divided by submitted volume. |
| `trade_active_imbalance` | `(active buy volume - active sell volume) / active trade volume`. |
| `cancel_buy_sell_imbalance` | `(buy cancel volume - sell cancel volume) / cancel volume`. |
| `trade_missing_bs_flag_volume` | Ordinary trade volume whose raw `bs_flag` is missing. |
| `trade_inferred_aggressor_volume` | Missing-`bs_flag` trade volume repaired by comparing ask/bid order submit times. |
| `trade_unresolved_aggressor_volume` | Missing-`bs_flag` trade volume that still cannot be assigned. |

Aggressor-side repair rule: when an ordinary trade lacks `bs_flag`, compare the referenced ask and bid order submit times; the later order is treated as the aggressor. Ties remain unresolved. In `20170123 / 000725.SZ`, ordinary trade `bs_flag` was already fully observed, so repair volume is zero.

## Minute Quality Gates

Command:

```bash
alpha-lab a-share-l3-quality-gates --dates 20170123,20170124 --wind-codes 000725.SZ,000001.SZ,000002.SZ
```

Default output:

`reports/l2_l3/minute_quality_gates.csv`

Gate status:

| Status | Meaning |
| --- | --- |
| `pass` | No hard failures and no unresolved aggressor-side repairs. |
| `review` | Hard accounting checks pass, but some soft issue needs inspection, currently unresolved aggressor side. |
| `fail` | At least one hard quality check failed. |

Hard checks:

| Check | Failure condition |
| --- | --- |
| Lifecycle remaining volume | Any order has negative remaining volume beyond tolerance. |
| Submit volume closure | Minute `submit_*` totals differ from lifecycle totals. |
| Event volume closure | Minute trade/cancel totals differ from raw trade/cancel totals. |
| Quote volume closure | Raw ordinary trade volume differs from quote cumulative volume. |
| Ratio bounds | Fill/cancel/open ratios outside `[0, 1]`. |
| Imbalance bounds | Submit/trade/cancel imbalances outside `[-1, 1]`. |
| Critical nulls | Critical numeric columns contain nulls after aggregation. |

## Coverage Report

Command:

```bash
alpha-lab a-share-l3-coverage
```

Default output:

`reports/l2_l3/daily_coverage.csv`

Summary over the 11 complete local days:

| Metric | Min | Max | Mean |
| --- | ---: | ---: | ---: |
| SZ stocks with order stream | 1,751 | 2,115 | 1,912.1 |
| Stocks with trade stream | 2,879 | 3,553 | 3,189.5 |
| Trade-stream stocks without orders | 1,128 | 1,438 | 1,277.5 |
| Ask fill hit rate for stocks with orders | 100.0% | 100.0% | 100.0% |
| Bid fill hit rate for stocks with orders | 100.0% | 100.0% | 100.0% |
| Cancel hit rate for stocks with orders | 100.0% | 100.0% | 100.0% |

Total scanned rows:

| Stream slice | Rows |
| --- | ---: |
| Orders | 173,071,637 |
| Trades/cancels, all exchanges | 227,444,125 |
| Trades/cancels for stocks with orders | 150,026,422 |

Interpretation: the lifecycle pipeline should be scoped to SZ stocks with order streams. SH names are visible in quotes/trades but do not have order streams in this local dataset.

## 000725.SZ Single-Stock Smoke Test

`20170123 / 000725.SZ` produced:

| Metric | Value |
| --- | ---: |
| Orders | 175,715 |
| Trades | 142,380 |
| Executed trades | 105,637 |
| Cancel events | 36,743 |
| Ask fill hit rate | 100.0% |
| Bid fill hit rate | 100.0% |
| Cancel hit rate | 100.0% |
| Negative remaining orders | 0 |
| Terminal observed orders | 142,569 |
| Open orders | 33,146 |
| Trade volume vs quote `cum_volume` diff | 0 |

Terminal-state counts:

| State | Orders |
| --- | ---: |
| `FULLY_FILLED` | 105,826 |
| `FULLY_CANCELED` | 36,701 |
| `UNFILLED_OPEN` | 33,138 |
| `PARTIAL_FILLED_CANCELED` | 42 |
| `PARTIAL_OPEN` | 8 |

Minute smoke-test totals:

| Metric | Value |
| --- | ---: |
| Minute rows | 254 |
| Feature columns | 51 |
| Submit volume diff vs lifecycle | 0 |
| Submit filled/canceled/remaining diff vs lifecycle | 0 |
| Trade event volume diff vs quotes | 0 |
| Cancel event volume diff vs lifecycle | 0 |
| Missing ordinary-trade `bs_flag` volume | 0 |

Quality-gate sample:

`20170123,20170124 × 000725.SZ,000001.SZ,000002.SZ` produced 6/6 `pass` rows. All accounting diffs were 0, ratio/imbalance out-of-bounds counts were 0, and missing ordinary-trade `bs_flag` volume was 0.

## Static Force Clustering

Command:

```bash
alpha-lab a-share-l3-static-clusters --dates 20170123,20170124 --wind-codes 000725.SZ,000001.SZ,000002.SZ
```

Default output directory:

`reports/l2_l3/static_clusters/`

Output files:

| File | Meaning |
| --- | --- |
| `labels.parquet` | Minute-level labels with raw clipped features, `z_*` standardized features, cluster id, and distance to assigned centroid. |
| `diagnostics.csv` | Candidate-K diagnostics: KMeans inertia and silhouette. |
| `profile.csv` | Cluster-level behavior profile in original feature units. |
| `shares.csv` | Long-form `date × wind_code × cluster` share table for stability checks. |

Default feature set:

`submit_buy_sell_imbalance`, `submit_order_hhi`, `submit_fill_ratio`, `submit_cancel_ratio`, `submit_open_ratio`, `submit_top_order_share`, `trade_active_imbalance`, `cancel_buy_sell_imbalance`, `submit_volume`, `trade_volume`, `event_cancel_volume`, `submit_mean_lifetime_lower_bound_ms`.

Preprocessing: volume/lifetime features are transformed with `log1p` for clustering, all features are winsorized by `clip_quantile`, then z-scored. Profiles report the winsorized original units, not the log-z units.

Sample result on `20170123,20170124 × 000725.SZ,000001.SZ,000002.SZ`:

| K | Silhouette |
| ---: | ---: |
| 4 | 0.2329 |
| 3 | 0.2304 |
| 2 | 0.2099 |
| 5 | 0.2012 |
| 6 | 0.1798 |

Selected `K=4`, but separation is weak and K=4 only narrowly beats K=3. Share diagnostics also show strong stock-level composition differences: for example, cluster 0 is dominated by `000725.SZ`, while cluster 1 is dominated by `000002.SZ`. Treat this as a working behavior taxonomy, not yet evidence of stable hidden forces.
