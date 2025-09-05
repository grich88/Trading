import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from historical_data_collector import fetch_historical_data, calculate_indicators
from enhanced_backtester import EnhancedBacktester
try:
    # Optional: Coinglass integration (used only if available)
    from coinglass_api import get_liquidation_heatmap, extract_liquidation_clusters
    HAS_COINGLASS = True
except Exception:
    HAS_COINGLASS = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_performance(results_df: pd.DataFrame, out_path: str) -> None:
    # Compute cum curves if not present
    df = results_df.copy()
    if "returns" not in df.columns:
        df["returns"] = df["close"].pct_change()
    if "strategy_returns" not in df.columns:
        df["position"] = df["signal_value"].shift(1).fillna(0).clip(-1, 1)
        df["strategy_returns"] = df["position"] * df["returns"]
    df["cum_bh"] = (1 + df["returns"]).cumprod() - 1
    df["cum_strat"] = (1 + df["strategy_returns"]).cumprod() - 1

    plt.figure(figsize=(10, 6))
    df[["cum_bh", "cum_strat"]].plot(ax=plt.gca())
    plt.title("Cumulative Returns: Buy&Hold vs Strategy")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_markdown_report(base_dir: str, run_ts: str, summary_rows: list, assets: list,
                          optimized_rows: list | None = None) -> None:
    report_path = os.path.join(base_dir, f"report_{run_ts}.md")
    lines = [
        f"# Enhanced Backtest Report ({run_ts})",
        "",
        "## Summary Metrics",
        "",
        "| Asset | Total Return | Buy&Hold | Sharpe | MaxDD | WinRate | Trades | TP1 Acc | TP2 Acc | SL Hit |",
        "|------:|------------:|--------:|------:|-----:|-------:|------:|--------:|--------:|-------:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['asset']} | {row['total_return']:.2%} | {row['buy_hold_return']:.2%} | "
            f"{row['sharpe']:.2f} | {row['max_drawdown']:.2%} | {row['win_rate']:.2%} | {row['total_trades']} | "
            f"{row['tp1_accuracy']:.2%} | {row['tp2_accuracy']:.2%} | {row['sl_hit_rate']:.2%} |"
        )
    # Optimization comparison table (baseline vs optimized cutoff)
    lines += [
        "",
        "## Optimization (Score Cutoff) Comparison",
        "",
        "| Asset | Label | Cutoff | WinRate → | Δ Win | TP1 Acc → | Δ TP1 | Sharpe → | Δ Sharpe | Total Ret → | Δ Ret |",
        "|------:|:-----:|------:|---------:|------:|----------:|------:|---------:|---------:|-------------:|------:|",
    ]
    for row in summary_rows:
        has_cut = ("cut_win_rate" in row)
        if not has_cut:
            continue
        win0, win1 = row.get('win_rate', 0.0), row.get('cut_win_rate', 0.0)
        tp10, tp11 = row.get('tp1_accuracy', 0.0), row.get('cut_tp1_accuracy', row.get('tp1_accuracy', 0.0))
        sh0, sh1 = row.get('sharpe', 0.0), row.get('cut_sharpe', row.get('sharpe', 0.0))
        r0, r1 = row.get('total_return', 0.0), row.get('cut_total_return', row.get('total_return', 0.0))
        lines.append(
            f"| {row['asset']} | {row.get('label','')} | {row.get('opt_cutoff',0.0):.3f} | "
            f"{win0:.2%} → {win1:.2%} | {(win1-win0):+.2%} | "
            f"{tp10:.2%} → {tp11:.2%} | {(tp11-tp10):+.2%} | "
            f"{sh0:.2f} → {sh1:.2f} | {(sh1-sh0):+.2f} | "
            f"{r0:.2%} → {r1:.2%} | {(r1-r0):+.2%} |"
        )
    lines += ["", "## Asset Plots", ""]
    for asset in assets:
        lines += [
            f"### {asset}",
            "",
            f"![{asset} performance]({asset}/{asset}_performance.png)",
            "",
        ]
    # If we have an optimized pass, add it
    if optimized_rows:
        lines += [
            "",
            "## Optimized Pass (Chosen Parameters)",
            "",
            "| Asset | Window | Lookahead | Cutoff | WinRate | Sharpe | Total Ret |",
            "|------:|------:|---------:|------:|-------:|------:|---------:|",
        ]
        for r in optimized_rows:
            lines.append(
                f"| {r['asset']} | {r['window']} | {r['lookahead']} | {r['cutoff']:.3f} | "
                f"{r['win_rate']:.2%} | {r['sharpe']:.2f} | {r['total_return']:.2%} |"
            )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved report: {report_path}")


def optimize_entry_threshold(results_df: pd.DataFrame, percentiles=(0, 25, 50, 60, 70, 80, 90)) -> dict:
    """Find absolute-score cutoff that maximizes TP1 accuracy while keeping signals."""
    df = results_df.copy()
    df = df[df['signal'].notna()]
    abs_scores = np.abs(df['score'].fillna(0).values)
    candidates = [np.percentile(abs_scores, p) for p in percentiles]
    best = {"cutoff": 0.0, "tp1_accuracy": 0.0, "signals": 0, "total_return": 0.0}
    for c in candidates:
        mask = np.abs(df['score']) >= c
        considered = df[mask]
        n = len(considered)
        if n == 0:
            continue
        tp1_acc = considered['hit_tp1'].mean()
        # approximate returns: zero out positions below cutoff
        tmp = df.copy()
        pos = tmp['signal_value'].fillna(0).clip(-1, 1)
        pos[~mask] = 0
        rets = tmp['close'].pct_change().fillna(0)
        strat = (pos.shift(1).fillna(0) * rets)
        tot = (1 + strat).cumprod().iloc[-1] - 1
        if tp1_acc > best["tp1_accuracy"] or (tp1_acc == best["tp1_accuracy"] and tot > best["total_return"]):
            best = {"cutoff": float(c), "tp1_accuracy": float(tp1_acc), "signals": int(n), "total_return": float(tot)}
    return best


def grid_search_params(df: pd.DataFrame, asset: str, static_liq: dict | None,
                       cutoff: float | None = None,
                       windows=(40, 60, 80), lookaheads=(10, 20, 30)) -> dict:
    """Grid-search window_size, lookahead, and optional cutoff; maximize win_rate then Sharpe."""
    from enhanced_backtester import EnhancedBacktester

    best = {"win_rate": -1.0, "sharpe": -1.0}
    tried = []
    for w in windows:
        for lh in lookaheads:
            for co in ([0.0, cutoff] if cutoff else [0.0]):
                bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=static_liq)
                bt.run_backtest(window_size=w, lookahead_candles=lh, min_abs_score=float(co or 0.0))
                m = bt.performance()
                tried.append({"window": w, "lookahead": lh, "cutoff": float(co or 0.0), **m})
                better = False
                if m.get("win_rate", 0.0) > best.get("win_rate", -1.0):
                    better = True
                elif m.get("win_rate", 0.0) == best.get("win_rate", -1.0) and m.get("sharpe", 0.0) > best.get("sharpe", -1.0):
                    better = True
                if better:
                    best = {"window": w, "lookahead": lh, "cutoff": float(co or 0.0), **m}
    best["trials"] = len(tried)
    return best


def run_pass(base_dir: str, run_ts: str, start_date: str, end_date: str, label: str) -> pd.DataFrame:
    summary = []
    assets_in_run = []

    for symbol, asset in [("BTC/USDT", "BTC"), ("SOL/USDT", "SOL")]:
        print(f"\n=== {symbol} ({label}) ===")
        df = fetch_historical_data(symbol, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            print("No data fetched");
            continue
        df = calculate_indicators(df)

        # Optional liquidation (best-effort)
        static_liq = None
        if HAS_COINGLASS:
            try:
                heatmap = get_liquidation_heatmap(asset)
                if heatmap:
                    static_liq = extract_liquidation_clusters(heatmap, df['close'].values)
            except Exception as e:
                print(f"Coinglass unavailable for {asset}: {e}")

        bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=static_liq)
        bt.run_backtest(window_size=60, lookahead_candles=20)
        metrics = bt.performance()
        print(metrics)

        # Output dirs
        pass_dir = os.path.join(base_dir, label)
        asset_dir = os.path.join(pass_dir, asset)
        ensure_dir(asset_dir)

        # Save results/metrics
        res_csv = os.path.join(asset_dir, f"{asset}_results.csv")
        bt.results.to_csv(res_csv)
        with open(os.path.join(asset_dir, f"{asset}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Optimization: percentile cutoff
        best = optimize_entry_threshold(bt.results)
        # Re-run with optimal cutoff for comparison
        if best.get("cutoff", 0.0) > 0:
            bt_cut = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=static_liq)
            bt_cut.run_backtest(window_size=60, lookahead_candles=20, min_abs_score=best["cutoff"])
            cut_metrics = bt_cut.performance()
            with open(os.path.join(asset_dir, f"{asset}_metrics_cutoff.json"), "w", encoding="utf-8") as f:
                json.dump(cut_metrics, f, indent=2)
            bt_cut.results.to_csv(os.path.join(asset_dir, f"{asset}_results_cutoff.csv"))
            plot_performance(bt_cut.results, os.path.join(asset_dir, f"{asset}_performance_cutoff.png"))
        else:
            cut_metrics = {}

        # Param grid search
        grid_best = grid_search_params(df, asset, static_liq, best.get("cutoff"))
        with open(os.path.join(asset_dir, f"{asset}_grid_search_best.json"), "w", encoding="utf-8") as f:
            json.dump(grid_best, f, indent=2)
        with open(os.path.join(asset_dir, f"{asset}_threshold_optimization.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

        # Plot
        plot_performance(bt.results, os.path.join(asset_dir, f"{asset}_performance.png"))

        row = metrics.copy(); row["asset"] = asset; row["label"] = label; row.update({
            "opt_cutoff": best.get("cutoff", 0.0),
            "opt_tp1_acc": best.get("tp1_accuracy", 0.0),
            "opt_signals": best.get("signals", 0),
            "opt_total_return": best.get("total_return", 0.0),
        })
        if cut_metrics:
            row.update({
                "cut_total_return": cut_metrics.get("total_return", 0.0),
                "cut_sharpe": cut_metrics.get("sharpe", 0.0),
                "cut_win_rate": cut_metrics.get("win_rate", 0.0),
                "cut_tp1_accuracy": cut_metrics.get("tp1_accuracy", 0.0),
                "cut_tp2_accuracy": cut_metrics.get("tp2_accuracy", 0.0),
                "cut_total_trades": cut_metrics.get("total_trades", 0),
            })
        if grid_best:
            row.update({
                "grid_win_rate": grid_best.get("win_rate", 0.0),
                "grid_sharpe": grid_best.get("sharpe", 0.0),
                "grid_total_return": grid_best.get("total_return", 0.0),
                "grid_window": grid_best.get("window", 0),
                "grid_lookahead": grid_best.get("lookahead", 0),
                "grid_cutoff": grid_best.get("cutoff", 0.0),
            })
        summary.append(row); assets_in_run.append(asset)

        print(f"Saved results to: {res_csv}")

    return pd.DataFrame(summary)


def choose_best_params(merged: pd.DataFrame) -> dict:
    """Select per-asset params that maximize predictive power (win_rate, then Sharpe)."""
    best: dict[str, dict] = {}
    assets = sorted(set(merged["asset"])) if "asset" in merged.columns else []
    for asset in assets:
        sub = merged[(merged["asset"] == asset) & merged["grid_window"].notna()]
        if sub.empty:
            continue
        sub = sub.sort_values(["grid_win_rate", "grid_sharpe"], ascending=[False, False])
        top = sub.iloc[0]
        best[asset] = {
            "window": int(top.get("grid_window", 60)),
            "lookahead": int(top.get("grid_lookahead", 20)),
            "cutoff": float(top.get("grid_cutoff", 0.0)),
        }
    return best


def run_optimized_pass(base_dir: str, run_ts: str, best_params: dict) -> list:
    """Run a final optimized pass using chosen params; return rows for the report."""
    rows: list[dict] = []
    opt_dir = os.path.join(base_dir, "optimized")
    ensure_dir(opt_dir)
    for asset, bp in best_params.items():
        symbol = f"{asset}/USDT"
        print(f"\n=== {symbol} (optimized defaults) ===")
        df = fetch_historical_data(symbol, start_date="2023-01-01", end_date="2025-08-01")
        if df is None or df.empty:
            continue
        df = calculate_indicators(df)
        static_liq = None
        if HAS_COINGLASS:
            try:
                heatmap = get_liquidation_heatmap(asset)
                if heatmap:
                    static_liq = extract_liquidation_clusters(heatmap, df['close'].values)
            except Exception as e:
                print(f"Coinglass unavailable for {asset}: {e}")
        bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=static_liq)
        bt.run_backtest(window_size=bp["window"], lookahead_candles=bp["lookahead"], min_abs_score=bp["cutoff"])
        m = bt.performance()
        asset_dir = os.path.join(opt_dir, asset)
        ensure_dir(asset_dir)
        bt.results.to_csv(os.path.join(asset_dir, f"{asset}_results_opt.csv"))
        with open(os.path.join(asset_dir, f"{asset}_metrics_opt.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        plot_performance(bt.results, os.path.join(asset_dir, f"{asset}_performance_opt.png"))
        rows.append({
            "asset": asset,
            "window": bp["window"],
            "lookahead": bp["lookahead"],
            "cutoff": bp["cutoff"],
            "win_rate": m.get("win_rate", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "total_return": m.get("total_return", 0.0),
        })
    # Persist chosen params
    with open(os.path.join(opt_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    return rows


def main() -> None:
    run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("data", "backtests", run_ts)
    ensure_dir(base_dir)

    # Full-period pass (as before)
    full_df = run_pass(base_dir, run_ts, start_date="2023-01-01", end_date="2025-08-01", label="full_period")

    # Last-1-year pass
    now = datetime.utcnow().date()
    last_year_start = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    last_year_end = now.strftime("%Y-%m-%d")
    last_df = run_pass(base_dir, run_ts, start_date=last_year_start, end_date=last_year_end, label="last_year")

    # Side-by-side comparison
    if not full_df.empty or not last_df.empty:
        merged = pd.concat([full_df, last_df], ignore_index=True)
        merged_csv = os.path.join(base_dir, f"comparison_{run_ts}.csv")
        merged.to_csv(merged_csv, index=False)
        print(f"Saved comparison: {merged_csv}")

        # Build combined markdown report with plots
        # Collect unique assets seen across both runs
        assets = sorted(set(merged["asset"])) if "asset" in merged.columns else ["BTC", "SOL"]
        # choose best params per asset and run final optimized pass
        best_params = choose_best_params(merged)
        opt_rows = run_optimized_pass(base_dir, run_ts, best_params) if best_params else []
        write_markdown_report(base_dir, run_ts, merged.to_dict(orient="records"), assets, opt_rows)


if __name__ == "__main__":
    main()