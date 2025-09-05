import os
from datetime import datetime
import pandas as pd

from historical_data_collector import fetch_historical_data, calculate_indicators
from enhanced_backtester import EnhancedBacktester


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def analyze_asset(symbol: str, asset: str, window: int, lookahead: int, cutoff: float) -> dict:
    df = fetch_historical_data(symbol, timeframe="4h", start_date=None, end_date=None)
    if df is None or df.empty:
        return {"asset": asset, "error": "no_data"}
    df = calculate_indicators(df)
    bt = EnhancedBacktester(df, asset_type=asset)
    bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff)
    results = bt.results
    last = results.iloc[-1]
    metrics = bt.performance()
    return {
        "asset": asset,
        "price": float(last["close"]),
        "signal": last["signal"],
        "score": float(last.get("score", 0.0)),
        "tp1": float(last.get("TP1", 0.0)),
        "tp2": float(last.get("TP2", 0.0)),
        "sl": float(last.get("SL", 0.0)),
        "rsi": float(last.get("rsi_raw", 0.0)),
        "metrics": metrics,
    }


def advise_position(row: dict) -> str:
    sig = row.get("signal")
    price = row.get("price")
    tp1, tp2, sl = row.get("tp1"), row.get("tp2"), row.get("sl")
    if sig in ("BUY", "STRONG BUY"):
        risk = max(0.0001, abs(price - sl) / price)
        text = (
            f"LONG {row['asset']}: entry ~ {price:.2f}, TP1 {tp1:.2f}, TP2 {tp2:.2f}, SL {sl:.2f}. "
            f"Signal={sig}, score={row.get('score',0):.3f}, RSI={row.get('rsi',0):.1f}. Risk ~{risk:.2%}."
        )
    elif sig in ("SELL", "STRONG SELL"):
        risk = max(0.0001, abs(sl - price) / price)
        text = (
            f"SHORT {row['asset']}: entry ~ {price:.2f}, TP1 {tp1:.2f}, TP2 {tp2:.2f}, SL {sl:.2f}. "
            f"Signal={sig}, score={row.get('score',0):.3f}, RSI={row.get('rsi',0):.1f}. Risk ~{risk:.2%}."
        )
    else:
        text = (
            f"NEUTRAL {row['asset']}: price {price:.2f}. No position. Last score={row.get('score',0):.3f}, "
            f"RSI={row.get('rsi',0):.1f}."
        )
    return text


def main() -> None:
    # Optimized defaults from latest run
    params = {
        "BTC": {"symbol": "BTC/USDT", "window": 60, "lookahead": 10, "cutoff": 0.425},
        "SOL": {"symbol": "SOL/USDT", "window": 40, "lookahead": 10, "cutoff": 0.520},
    }
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data", "live_signals", ts)
    ensure_dir(out_dir)

    rows = []
    for asset, p in params.items():
        print(f"\nFetching and analyzing {asset}...")
        res = analyze_asset(p["symbol"], asset, p["window"], p["lookahead"], p["cutoff"])
        rows.append(res)
        print(advise_position(res))

    # Save snapshot
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "signals.csv"), index=False)
    with open(os.path.join(out_dir, "signals.json"), "w", encoding="utf-8") as f:
        f.write(df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()


