
# Analysis Report – BONK

## Parameters
- Window: 40
- Lookahead candles: 18
- Score cutoff (abs): 0.450
- Data range: 2023-01-01 → 2025-08-25
- Coinglass used: Yes

## Current Snapshot
- Price: 0.00
- Signal: NEUTRAL
- Score: -0.066 → adjusted -0.065
- RSI: 47.39
- ATR%: 3.49%
- Targets: TP1 2.3e-05, TP2 2.3e-05, SL 2.2e-05
- Leverage suggestion: 1x

## Rationale
- RSI +0.00
- Volume +0.00
- Divergence +0.00
- WebTrend -0.30
- Liquidity -0.80

### TradingView OCR

- Bias applied: -0.06 (15% blend)

| Panel | Weight | Score | Buy | Sell | Neutral |
|---|---:|---:|---:|---:|---:|
| oscillators | 0.20 | +0.17 | 11 | 8 | 7 |
| oscillators | 0.20 | +0.88 | 14 | 0 | 10 |
| unknown | 0.10 | +0.00 | 0 | 0 | 0 |
| oscillators | 0.20 | -0.17 | 8 | 11 | 7 |
| oscillators | 0.20 | -0.54 | 3 | 12 | 10 |
| pivots | 0.15 | +0.00 | 0 | 0 | 0 |
| oscillators | 0.20 | -0.17 | 9 | 12 | 6 |
| oscillators | 0.20 | -0.62 | 2 | 12 | 10 |
| unknown | 0.10 | +0.00 | 0 | 0 | 0 |

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
- total_return: -16.97%
- buy_hold_return: -28.85%
- sharpe: -0.084
- max_drawdown: 41.85%
- win_rate: 47.83%
- total_trades: 69
- tp1_accuracy: 1.00%
- tp2_accuracy: 0.65%
- sl_hit_rate: 2.16%

## Assets
- Results CSV: data\ui_exports\20250824_133101\BONK\BONK_results.csv
- Price & Signals: data\ui_exports\20250824_133101\BONK\BONK_price_signals.png
- Cumulative: data\ui_exports\20250824_133101\BONK\BONK_cumulative.png

## Advisory
- Negative edge: raise cutoff by 0.05, enable regime filter, or increase window to smooth noise.
- TP1 hit rate is low within lookahead. Increase lookahead by 5 or move TP1 closer (e.g., 3–3.5% for longs).
