
# Analysis Report – SOL

## Parameters
- Window: 35
- Lookahead candles: 10
- Score cutoff (abs): 0.460
- Data range: 2023-01-01 → 2025-08-24
- Coinglass used: No

## Current Snapshot
- Price: 203.30
- Signal: NEUTRAL
- Score: +0.240
- RSI: 65.93
- ATR%: 2.72%
- Targets: TP1 207.37, TP2 211.43, SL 199.23
- Leverage suggestion: 1x

## Rationale
- RSI +0.80
- Volume +0.00
- Divergence +0.00
- WebTrend +0.30
- Liquidity +0.00

## How to interpret
- Score combines features into [-1,1]; entries are taken when |score| ≥ cutoff (and pass the regime filter).

- Regime filter trades with the trend (EMA stack). If a signal conflicts with trend, it will be set to NEUTRAL.

- Adaptive cutoff raises selectivity in high ATR to avoid low‑quality trades.

- TP/SL are ATR‑based and adjusted toward liquidation clusters and pivots to respect likely reaction zones.

- Fees are charged at position changes using the configured bps.

## Tuning guidance
- Too few trades: decrease cutoff by ~0.05 or reduce window by 5–10.

- Too noisy/negative Sharpe: raise cutoff by ~0.05, increase window, and keep regime filter on.

- Low TP hits: increase lookahead or tighten TP1 slightly.

- High ATR: reduce leverage, keep adaptive cutoff on.

## Performance (backtest on selected range)
- total_return: 50.56%
- buy_hold_return: 1939.12%
- sharpe: 0.445
- max_drawdown: 18.81%
- win_rate: 60.69%
- total_trades: 145
- tp1_accuracy: 1.81%
- tp2_accuracy: 0.81%
- sl_hit_rate: 2.21%

## Assets
- Results CSV: data\ui_exports\20250823_163102\SOL\SOL_results.csv
- Price & Signals: data\ui_exports\20250823_163102\SOL\SOL_price_signals.png
- Cumulative: data\ui_exports\20250823_163102\SOL\SOL_cumulative.png

## Advisory
- Upload TradingView panels or paste OCR text to add external bias and context.
- Liquidation clusters not used. Enable Coinglass (when endpoint responds) to refine TP/SL levels.
- TP1 hit rate is low within lookahead. Increase lookahead by 5 or move TP1 closer (e.g., 3–3.5% for longs).
