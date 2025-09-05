
# Analysis Report – SOL

## Parameters
- Window: 40
- Lookahead candles: 40
- Score cutoff (abs): 0.520
- Data range: 2023-01-01 → 2025-08-25
- Coinglass used: No

## Current Snapshot
- Price: 204.09
- Signal: NEUTRAL
- Score: +0.130 → adjusted 0.143
- RSI: 62.19
- ATR%: 2.87%
- Targets: TP1 208.17, TP2 212.25, SL 200.01
- Leverage suggestion: 1x

## Rationale
- RSI +0.80
- Volume +0.00
- Divergence +0.00
- WebTrend +0.30
- Liquidity -0.80

### TradingView OCR

- Bias applied: +0.22 (15% blend)

| Panel | Weight | Score | Buy | Sell | Neutral |
|---|---:|---:|---:|---:|---:|
| moving_averages | 0.25 | +0.00 | 0 | 0 | 0 |
| unknown | 0.10 | +0.00 | 0 | 0 | 0 |
| oscillators | 0.20 | -0.06 | 9 | 10 | 7 |
| oscillators | 0.20 | +0.50 | 5 | 0 | 10 |
| unknown | 0.10 | +0.00 | 0 | 0 | 0 |
| oscillators | 0.20 | +0.17 | 12 | 9 | 6 |
| oscillators | 0.20 | +0.92 | 16 | 0 | 10 |
| pivots | 0.15 | +0.00 | 0 | 0 | 0 |

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
- total_return: 75.53%
- buy_hold_return: 1947.04%
- sharpe: 0.585
- max_drawdown: 15.28%
- win_rate: 64.47%
- total_trades: 76
- tp1_accuracy: 0.10%
- tp2_accuracy: 0.86%
- sl_hit_rate: 1.43%

## Assets
- Results CSV: data\ui_exports\20250824_112818\SOL\SOL_results.csv
- Price & Signals: data\ui_exports\20250824_112818\SOL\SOL_price_signals.png
- Cumulative: data\ui_exports\20250824_112818\SOL\SOL_cumulative.png

## Advisory
- Liquidation clusters not used. Enable Coinglass (when endpoint responds) to refine TP/SL levels.
- TP1 hit rate is low within lookahead. Increase lookahead by 5 or move TP1 closer (e.g., 3–3.5% for longs).
