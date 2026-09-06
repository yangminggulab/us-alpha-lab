# L2 / L3 Data Inventory

## Raw Dataset

- Location: `../../data/raw/a_share_l2_hf/`
- Streams per complete day:
  - `行情.parquet`: 10 档盘口快照。
  - `逐笔委托.parquet`: 逐笔委托。
  - `逐笔成交.parquet`: 逐笔成交。
- Local size after the latest download: about 8.1G.
- Disk free after the latest download: about 55Gi.

## Complete Local Trading Days

The local dataset currently has 11 complete trading days:

| Date | Status |
| --- | --- |
| 20170123 | complete |
| 20170124 | complete |
| 20170125 | complete |
| 20170206 | complete |
| 20170515 | complete |
| 20170606 | complete |
| 20180829 | complete |
| 20181211 | complete |
| 20181219 | complete |
| 20181226 | complete |
| 20190102 | complete |

## Validation Notes

- No `.part` files were left after download.
- All listed parquet files were readable via parquet metadata.
- This is enough for pipeline prototyping and early signal checks, but not enough for production-grade conclusions.
- Coverage scan output: `../../reports/l2_l3/daily_coverage.csv`.
- Across the 11 local days, order streams cover SZ stocks only; stocks with order streams have 100% ask/bid fill id hit rates and 100% cancel id hit rates against `ex_order_id`.
