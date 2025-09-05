
# Analysis Report – SOL

## Parameters
- Window: 40
- Lookahead candles: 40
- Score cutoff (abs): 0.375
- Data range: 2024-08-23 → 2025-08-23
- Coinglass used: No

## Current Snapshot
- Price: 198.16
- Signal: BUY
- Score: +0.570
- RSI: 64.27
- ATR%: 2.50%
- Targets: TP1 220.79, TP2 228.77, SL 193.78
- Leverage suggestion: 3x

## Rationale
- RSI +0.80
- Volume +1.00
- Divergence +0.00
- WebTrend +0.30
- Liquidity +0.80

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
- total_return: 79.81%
- buy_hold_return: 37.15%
- sharpe: 0.898
- max_drawdown: 27.50%
- win_rate: 59.81%
- total_trades: 107
- tp1_accuracy: 5.02%
- tp2_accuracy: 2.78%
- sl_hit_rate: 7.94%

## Assets
- Results CSV: data\ui_exports\20250823_195759\SOL\SOL_results.csv
- Price & Signals: data\ui_exports\20250823_195759\SOL\SOL_price_signals.png
- Cumulative: data\ui_exports\20250823_195759\SOL\SOL_cumulative.png

## Advisory
- Upload TradingView panels or paste OCR text to add external bias and context.
- Liquidation clusters not used. Enable Coinglass (when endpoint responds) to refine TP/SL levels.
