
# Analysis Report – BTC

## Parameters
- Window: 60
- Lookahead candles: 17
- Score cutoff (abs): 0.425
- Data range: 2024-08-23 → 2025-08-23
- Coinglass used: No

## Current Snapshot
- Price: 115568.77
- Signal: NEUTRAL
- Score: +0.084
- RSI: 54.59
- ATR%: 1.13%
- Targets: TP1 117880.15, TP2 120191.52, SL 113257.39
- Leverage suggestion: 1x

## Rationale
- RSI +0.00
- Volume +1.00
- Divergence +0.00
- WebTrend -0.30
- Liquidity -0.80

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
- total_return: -11.56%
- buy_hold_return: 90.73%
- sharpe: -0.531
- max_drawdown: 14.32%
- win_rate: 51.35%
- total_trades: 37
- tp1_accuracy: 0.09%
- tp2_accuracy: 0.05%
- sl_hit_rate: 2.19%

## Assets
- Results CSV: data\ui_exports\20250823_195523\BTC\BTC_results.csv
- Price & Signals: data\ui_exports\20250823_195523\BTC\BTC_price_signals.png
- Cumulative: data\ui_exports\20250823_195523\BTC\BTC_cumulative.png

## Advisory
- Upload TradingView panels or paste OCR text to add external bias and context.
- Liquidation clusters not used. Enable Coinglass (when endpoint responds) to refine TP/SL levels.
- TP1 hit rate is low within lookahead. Increase lookahead by 5 or move TP1 closer (e.g., 3–3.5% for longs).
