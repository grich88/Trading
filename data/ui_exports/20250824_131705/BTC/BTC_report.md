
# Analysis Report – BTC

## Parameters
- Window: 60
- Lookahead candles: 19
- Score cutoff (abs): 0.425
- Data range: 2023-01-01 → 2025-08-25
- Coinglass used: Yes

## Current Snapshot
- Price: 114676.47
- Signal: NEUTRAL
- Score: -0.016 → adjusted -0.022
- RSI: 48.49
- ATR%: 0.97%
- Targets: TP1 116970.0, TP2 119263.53, SL 112382.94
- Leverage suggestion: 1x

## Rationale
- RSI +0.00
- Volume +0.00
- Divergence +0.50
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
- total_return: 43.73%
- buy_hold_return: 593.62%
- sharpe: 0.522
- max_drawdown: 23.17%
- win_rate: 55.24%
- total_trades: 105
- tp1_accuracy: 0.00%
- tp2_accuracy: 0.00%
- sl_hit_rate: 2.71%

## Assets
- Results CSV: data\ui_exports\20250824_131705\BTC\BTC_results.csv
- Price & Signals: data\ui_exports\20250824_131705\BTC\BTC_price_signals.png
- Cumulative: data\ui_exports\20250824_131705\BTC\BTC_cumulative.png

## Advisory
- TP1 hit rate is low within lookahead. Increase lookahead by 5 or move TP1 closer (e.g., 3–3.5% for longs).
