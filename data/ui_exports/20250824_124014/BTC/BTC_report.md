
# Analysis Report – BTC

## Parameters
- Window: 60
- Lookahead candles: 10
- Score cutoff (abs): 0.200
- Data range: 2024-08-24 → 2025-08-24
- Coinglass used: No

## Current Snapshot
- Price: 115036.00
- Signal: NEUTRAL
- Score: +0.148 → adjusted 0.176
- RSI: 50.86
- ATR%: 1.04%
- Targets: TP1 117336.72, TP2 119637.44, SL 112735.28
- Leverage suggestion: 1x

## Rationale
- RSI +0.80
- Volume +0.00
- Divergence +0.00
- WebTrend -0.30
- Liquidity -0.80

### TradingView OCR

- Bias applied: +0.34 (15% blend)

| Panel | Weight | Score | Buy | Sell | Neutral |
|---|---:|---:|---:|---:|---:|
| oscillators | 0.20 | -0.06 | 9 | 10 | 7 |
| oscillators | 0.20 | +0.35 | 3 | 0 | 10 |
| pivots | 0.15 | +0.00 | 0 | 0 | 0 |
| oscillators | 0.20 | +0.17 | 12 | 9 | 6 |
| oscillators | 0.20 | +0.92 | 16 | 0 | 10 |
| unknown | 0.10 | +0.00 | 0 | 0 | 0 |
| oscillators | 0.20 | +0.17 | 12 | 9 | 6 |
| oscillators | 0.20 | +0.92 | 16 | 0 | 10 |
| unknown | 0.10 | +0.00 | 0 | 0 | 0 |
| oscillators | 0.20 | +0.17 | 11 | 8 | 7 |
| oscillators | 0.20 | +0.88 | 14 | 0 | 10 |
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
- total_return: 32.25%
- buy_hold_return: 80.22%
- sharpe: 0.692
- max_drawdown: 17.43%
- win_rate: 63.93%
- total_trades: 122
- tp1_accuracy: 0.32%
- tp2_accuracy: 0.00%
- sl_hit_rate: 6.53%

## Assets
- Results CSV: data\ui_exports\20250824_124014\BTC\BTC_results.csv
- Price & Signals: data\ui_exports\20250824_124014\BTC\BTC_price_signals.png
- Cumulative: data\ui_exports\20250824_124014\BTC\BTC_cumulative.png

## Advisory
- Liquidation clusters not used. Enable Coinglass (when endpoint responds) to refine TP/SL levels.
- TP1 hit rate is low within lookahead. Increase lookahead by 5 or move TP1 closer (e.g., 3–3.5% for longs).
