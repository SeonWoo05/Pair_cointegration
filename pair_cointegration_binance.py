
"""
pair_cointegration_binance.py

Purpose:
- Fetch BTCUSDT and the top 9 non-BTC altcoins by 24h quote volume (USDT quote) from Binance.
- For each of three timeframes (1m, 1h, 1d) with respective lookback lengths (1000, 500, 100),
  compute pairwise cointegration tests between X(t)=BTCUSDT and Y(t+k) for lags k in {3,5,10}.
- Report Pearson correlation, Engle-Granger cointegration (statsmodels.coint), and ADF of residuals.
- Save results to CSV and (optionally) plots.

Notes:
- Requires internet access to call Binance public REST API (no keys needed).
- Uses: pandas, numpy, requests, statsmodels, scipy

CLI:
    python pair_cointegration_binance.py --run

    Optional arguments:
        --asof "2025-08-25 00:00:00"  # end time (UTC) for data windows; default: now
        --quote USDT                   # quote asset for ranking and bars
        --lags 3 5 10                  # lags to test (default 3 5 10)
        --outdir ./results             # output directory for csv/plots
        --no-plots                     # skip plots
        --topn 9                       # number of altcoins to test (default 9)

Outputs:
- <outdir>/pair_cointegration_results_<timestamp>.csv
- <outdir>/top_alts_<timestamp>.csv
- (optional) <outdir>/plots/*.png

Author: ChatGPT (GPT-5 Pro)
"""

import os, sys, time, math, logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd

# Stats/statistical tests
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from scipy.stats import pearsonr

BINANCE_BASE = "https://api.binance.com"

# ---- Utilities ----

STABLE_BASES = {"USDT","USDC","BUSD","TUSD","FDUSD","DAI","UST","USTC"}

def ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def parse_asof(asof: Optional[str]) -> int:
    """Return endTime (ms since epoch) for Binance klines."""
    if not asof:
        return to_ms(datetime.utcnow())
    # try parse various formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(asof, fmt).replace(tzinfo=timezone.utc)
            return to_ms(dt)
        except ValueError:
            continue
    # if unix seconds
    try:
        val = int(asof)
        if val < 10_000_000_000:
            return val * 1000
        return val
    except Exception:
        pass
    raise ValueError(f"Unrecognized asof format: {asof}")

def http_get_json(url: str, params: Dict=None, retries: int=3, backoff: float=0.5):
    """GET JSON with basic retry/backoff."""
    params = params or {}
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            else:
                last_exc = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_exc = e
        time.sleep(backoff * (2**i))
    raise last_exc

# ---- Binance helpers ----

def get_top_alts_by_quote_volume(quote: str="USDT", topn: int=9, exclude_btc: bool=True,
                                 exclude_stables: bool=True) -> List[str]:
    """Return list of top N symbols (e.g., ETHUSDT) by 24h quote volume, excluding BTC and stables."""
    tickers = http_get_json(f"{BINANCE_BASE}/api/v3/ticker/24hr")
    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith(quote):
            continue
        base = sym[:-len(quote)]
        if exclude_btc and base == "BTC":
            continue
        if exclude_stables and base in STABLE_BASES:
            continue
        # skip leveraged tokens and weird products
        if any(x in base for x in ["UP","DOWN","BULL","BEAR"]):
            continue
        try:
            qv = float(t.get("quoteVolume", "0"))
        except Exception:
            qv = 0.0
        rows.append((sym, base, qv))
    rows.sort(key=lambda x: x[2], reverse=True)
    return [r[0] for r in rows[:topn]]

def get_klines(symbol: str, interval: str, limit: int, end_time_ms: Optional[int]=None) -> pd.DataFrame:
    """
    Fetch klines (candles) for a symbol.
    Returns DataFrame with columns: open_time, open, high, low, close, volume, close_time
    Index is datetime (open_time).
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    data = http_get_json(f"{BINANCE_BASE}/api/v3/klines", params=params)
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","n_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume","qav","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df[["close","volume","close_time"]]

# ---- Analysis helpers ----

@dataclass
class PairTestResult:
    timeframe: str
    n_obs: int
    lag_k: int
    symbol_y: str
    pearson_r: float
    pearson_p: float
    coint_t: float
    coint_p: float
    coint_crit_1pct: float
    coint_crit_5pct: float
    coint_crit_10pct: float
    adf_resid_stat: float
    adf_resid_p: float
    adf_resid_crit_1pct: float
    adf_resid_crit_5pct: float
    adf_resid_crit_10pct: float
    ols_alpha: float
    ols_beta: float
    n_used: int
    sample_start: str
    sample_end: str

def engle_granger_with_resid_adf(x: pd.Series, y: pd.Series) -> Tuple[dict, pd.Series, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    1) Engle-Granger cointegration test (statsmodels.coint) on x,y.
    2) OLS regression y ~ a + b*x, residuals e = y - a - b x
    3) ADF test on residuals.
    Returns (eg_res, resid, ols_fit)
    """
    # Ensure aligned index and drop NA
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 30:
        raise ValueError("Not enough overlapping observations for cointegration test (need >= 30).")
    # EG test
    tstat, pval, crit = coint(df["x"].values, df["y"].values, trend="c", maxlag=None, autolag="aic")
    # OLS
    X = sm.add_constant(df["x"].values)
    ols = sm.OLS(df["y"].values, X).fit()
    resid = df["y"].values - ols.predict(X)
    # ADF on residuals
    adf_res = adfuller(resid, autolag="AIC")
    eg_res = {
        "tstat": float(tstat),
        "pval": float(pval),
        "crit": {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
        "adf_stat": float(adf_res[0]),
        "adf_p": float(adf_res[1]),
        "adf_crit": {"1%": float(adf_res[4]["1%"]), "5%": float(adf_res[4]["5%"]), "10%": float(adf_res[4]["10%"])},
        "nused": int(adf_res[3]),
        "ols_alpha": float(ols.params[0]),
        "ols_beta": float(ols.params[1]),
    }
    return eg_res, pd.Series(resid, index=df.index), ols

def shift_align_for_lead(y: pd.Series, k: int) -> pd.Series:
    """
    Produce Y(t+k) aligned with X(t):
    In pandas terms: lead y by k => y.shift(-k)
    """
    return y.shift(-k)

def run_tests_for_symbol(
        btc_close: pd.Series,
        alt_close: pd.Series,
        timeframe: str,
        lag_list: List[int]
    ) -> List[PairTestResult]:
    results = []
    for k in lag_list:
        y_lead = shift_align_for_lead(alt_close, k)
        # Correlation
        df = pd.concat([btc_close.rename("x"), y_lead.rename("y")], axis=1).dropna()
        if len(df) < 30:
            continue
        r, rp = pearsonr(df["x"].values, df["y"].values)
        # Engle-Granger + ADF on residuals
        eg, resid, ols = engle_granger_with_resid_adf(df["x"], df["y"])
        res = PairTestResult(
            timeframe=timeframe,
            n_obs=len(df),
            lag_k=k,
            symbol_y=alt_close.name,
            pearson_r=float(r),
            pearson_p=float(rp),
            coint_t=eg["tstat"],
            coint_p=eg["pval"],
            coint_crit_1pct=eg["crit"]["1%"],
            coint_crit_5pct=eg["crit"]["5%"],
            coint_crit_10pct=eg["crit"]["10%"],
            adf_resid_stat=eg["adf_stat"],
            adf_resid_p=eg["adf_p"],
            adf_resid_crit_1pct=eg["adf_crit"]["1%"],
            adf_resid_crit_5pct=eg["adf_crit"]["5%"],
            adf_resid_crit_10pct=eg["adf_crit"]["10%"],
            ols_alpha=eg["ols_alpha"],
            ols_beta=eg["ols_beta"],
            n_used=eg["nused"],
            sample_start=df.index.min().strftime("%Y-%m-%d %H:%M:%S"),
            sample_end=df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
        )
        results.append(res)
    return results

def fetch_all_price_series(symbols: List[str], timeframes: Dict[str,int], asof_ms: Optional[int]) -> Dict[str, Dict[str, pd.Series]]:
    """
    Returns a nested dict: series_map[symbol][timeframe] = close price series (pd.Series with UTC index)
    Assumes BTCUSDT is also fetched separately.
    """
    series_map: Dict[str, Dict[str, pd.Series]] = {s: {} for s in symbols}
    for tf, limit in timeframes.items():
        for s in symbols:
            df = get_klines(s, tf, limit, asof_ms)
            series_map[s][tf] = df["close"].rename(s)
    return series_map

def run_experiment(
        asof: Optional[str]=None,
        quote: str="USDT",
        lags: List[int]=[3,5,10],
        timeframes: Dict[str,int] = {"1m":1000, "1h":500, "1d":100},
        topn: int=9,
        outdir: str="./results",
        make_plots: bool=True
    ) -> pd.DataFrame:
    """
    End-to-end experiment. Returns a DataFrame with all results and saves CSVs.
    """
    os.makedirs(outdir, exist_ok=True)
    asof_ms = parse_asof(asof) if asof else None

    # Universe
    top_alts = get_top_alts_by_quote_volume(quote=quote, topn=topn, exclude_btc=True, exclude_stables=True)
    # Ensure BTC symbol
    btc_sym = f"BTC{quote}"
    symbols = [btc_sym] + top_alts

    # Fetch series
    series_map = fetch_all_price_series(symbols, timeframes, asof_ms)

    # Prepare
    results: List[PairTestResult] = []
    for tf, limit in timeframes.items():
        btc_close = series_map[btc_sym][tf].copy()
        for alt in top_alts:
            alt_close = series_map[alt][tf].copy()
            # Align on common index
            idx = btc_close.index.intersection(alt_close.index)
            res_list = run_tests_for_symbol(btc_close.loc[idx], alt_close.loc[idx], tf, lags)
            results.extend(res_list)

    # To DataFrame
    res_df = pd.DataFrame([r.__dict__ for r in results])
    res_df["is_cointegrated_coint_p_5pct"] = res_df["coint_p"] < 0.05
    res_df["is_cointegrated_adf_p_5pct"] = res_df["adf_resid_p"] < 0.05
    res_df["evidence_any_5pct"] = res_df["is_cointegrated_coint_p_5pct"] | res_df["is_cointegrated_adf_p_5pct"]

    # Save
    stamp = ts()
    res_csv = os.path.join(outdir, f"pair_cointegration_results_{stamp}.csv")
    top_csv = os.path.join(outdir, f"top_alts_{stamp}.csv")
    res_df.to_csv(res_csv, index=False)
    pd.DataFrame({"symbol": top_alts}).to_csv(top_csv, index=False)

    # Optional plots: scatter X vs Y_lead for the strongest pair per timeframe
    if make_plots and len(res_df):
        import matplotlib.pyplot as plt
        for tf in sorted(res_df["timeframe"].unique()):
            sub = res_df[res_df["timeframe"]==tf].sort_values("coint_p")
            if len(sub)==0: 
                continue
            row = sub.iloc[0]
            k = int(row["lag_k"])
            sym = row["symbol_y"]
            btc = series_map[btc_sym][tf]
            alt = series_map[sym][tf]
            idx = btc.index.intersection(alt.index)
            df = pd.concat([btc.loc[idx].rename("x"), alt.loc[idx].rename("y")], axis=1).dropna()
            y_lead = df["y"].shift(-k)
            merged = pd.concat([df["x"], y_lead.rename("y_lead")], axis=1).dropna()
            if len(merged) < 30:
                continue
            plt.figure()
            plt.scatter(merged["x"].values, merged["y_lead"].values, s=6)
            plt.title(f"{tf} | {sym} vs BTC (Y lead k={k})")
            plt.xlabel("BTCUSDT close")
            plt.ylabel(f"{sym} close (lead {k})")
            plot_path = os.path.join(outdir, f"scatter_{tf}_{sym}_k{k}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

    return res_df

# ---- Synthetic verification (does not require internet) ----

def _random_walk(n, seed=None, sigma=1.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=sigma, size=n)
    return np.cumsum(e)

def _stationary_ar1(n, seed=None, phi=0.3, sigma=1.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=sigma, size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + e[t]
    return x

def synthetic_pair(n=1200, k=5, alpha=0.5, beta=1.7, seed=7):
    """
    Construct synthetic X and Y such that (X_t, Y_{t+k}) are cointegrated:
        X_t = RW
        Y_t = alpha + beta * X_{t-k} + u_t, where u_t stationary
        => Y_{t+k} = alpha + beta * X_t + u_{t+k}
    """
    X = _random_walk(n=n, seed=seed, sigma=1.0)
    u = _stationary_ar1(n=n, seed=seed+1, phi=0.5, sigma=0.8)
    Y = np.empty(n)
    Y[:] = np.nan
    for t in range(n):
        lag_index = t-k
        if lag_index >= 0:
            Y[t] = alpha + beta * X[lag_index] + u[t]
    # Build pandas series with a fake datetime index (1-min spacing)
    idx = pd.date_range("2025-01-01", periods=n, freq="T", tz="UTC")
    Xs = pd.Series(X, index=idx, name="BTCUSDT")
    Ys = pd.Series(Y, index=idx, name="ALTUSDT")
    return Xs, Ys

def synthetic_verify_all(lags=[3,5,10]) -> pd.DataFrame:
    """
    Run the core pipeline on synthetic data for 3 timeframes (using different lengths),
    verifying detection on known cointegrated and non-cointegrated pairs.
    """
    out = []
    # "Timeframes" with different lengths
    tf_map = {"1m":1000, "1h":500, "1d":100}
    for tf, n in tf_map.items():
        for k in lags:
            Xs, Ys = synthetic_pair(n=n+10, k=k, alpha=0.3, beta=1.4, seed=42+k)  # n+10 ensures enough after dropping
            X = Xs.iloc[:n]
            Y = Ys.iloc[:n]
            # Positive case: (X_t, Y_{t+k}) cointegrated by construction
            res_list = run_tests_for_symbol(X, Y.rename("ALTUSDT"), tf, [k])
            if res_list:
                r = res_list[0]
                out.append({
                    "timeframe": tf,
                    "lag_k": k,
                    "case": "cointegrated_true",
                    "pearson_r": r.pearson_r,
                    "coint_p": r.coint_p,
                    "adf_resid_p": r.adf_resid_p,
                    "ols_beta": r.ols_beta,
                    "n_obs": r.n_obs,
                })
            # Negative case: random independent RW for Y (should NOT be cointegrated with X_t)
            rng = np.random.default_rng(123+k)
            Y_bad = _random_walk(n=n+10, seed=88+k, sigma=1.0)
            idx = X.index
            Y_bad_s = pd.Series(Y_bad[:n], index=idx, name="BADALTUSDT")
            res_list2 = run_tests_for_symbol(X, Y_bad_s, tf, [k])
            if res_list2:
                r2 = res_list2[0]
                out.append({
                    "timeframe": tf,
                    "lag_k": k,
                    "case": "not_cointegrated_false",
                    "pearson_r": r2.pearson_r,
                    "coint_p": r2.coint_p,
                    "adf_resid_p": r2.adf_resid_p,
                    "ols_beta": r2.ols_beta,
                    "n_obs": r2.n_obs,
                })
    df = pd.DataFrame(out)
    return df

def main():
    import argparse
    p = argparse.ArgumentParser(description="BTC vs Top Altcoins (Binance) cointegration tests with Y lead (t+k).")
    p.add_argument("--run", action="store_true", help="Run the full experiment (requires internet).")
    p.add_argument("--asof", type=str, default=None, help="End time (UTC) for windows, e.g., '2025-08-25 00:00:00'.")
    p.add_argument("--quote", type=str, default="USDT", help="Quote asset (default USDT).")
    p.add_argument("--lags", type=int, nargs="+", default=[3,5,10], help="Lags k to test.")
    p.add_argument("--outdir", type=str, default="./results", help="Output directory.")
    p.add_argument("--no-plots", action="store_true", help="Disable scatter plots.")
    p.add_argument("--topn", type=int, default=9, help="Number of altcoins to test (default 9).")
    p.add_argument("--verify", action="store_true", help="Run synthetic verification only (no internet).")
    args = p.parse_args()

    if args.verify or not args.run:
        print("[Synthetic verification] Running pipeline checks without internet...")
        ver = synthetic_verify_all(lags=args.lags)
        stamp = ts()
        out_csv = os.path.join(args.outdir, f"synthetic_verification_{stamp}.csv")
        os.makedirs(args.outdir, exist_ok=True)
        ver.to_csv(out_csv, index=False)
        print(f"Saved synthetic verification to: {out_csv}")
        print(ver.head())
        if not args.run:
            return

    # If --run specified, execute the real experiment (internet required)
    print("[Real experiment] Fetching Binance data and running cointegration tests...")
    res_df = run_experiment(
        asof=args.asof,
        quote=args.quote,
        lags=args.lags,
        outdir=args.outdir,
        make_plots=(not args.no_plots),
        topn=args.topn
    )
    stamp = ts()
    out_csv = os.path.join(args.outdir, f"pair_cointegration_results_{stamp}.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")
    print(res_df.sort_values(["timeframe","lag_k","coint_p"]).groupby(["timeframe","lag_k"]).head(5))

if __name__ == "__main__":
    main()
