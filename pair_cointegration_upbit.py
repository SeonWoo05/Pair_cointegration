# -*- coding: utf-8 -*-
"""
pair_cointegration_upbit_converted_v3.py

- Upbit KRW-BTC vs Top KRW alts, cointegration with Y lead (k in {3,5,10})
- Strict effective sample sizes: 1m=1000, 1h=500, 1d=100
- Critical fix: paginate using UTC 'to' (oldest candle's candle_date_time_utc - 1s) to avoid duplicate pages.

Run:
  python pair_cointegration_upbit_converted_v3.py --run --outdir ./results
  python pair_cointegration_upbit_converted_v3.py --verify --outdir ./results
"""

import os, time, math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from scipy.stats import pearsonr

UPBIT_BASE = "https://api.upbit.com"
STABLE_BASES = {"USDT","USDC","BUSD","TUSD","FDUSD","DAI","UST","USTC","KRW"}

# ---------------- Utilities ----------------

def ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def parse_asof_utc(asof: Optional[str]) -> datetime:
    if not asof:
        return datetime.utcnow().replace(tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(asof, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        val = int(asof)
        if val < 10_000_000_000:   # seconds
            return datetime.fromtimestamp(val, tz=timezone.utc)
        return datetime.fromtimestamp(val/1000, tz=timezone.utc)  # ms
    except Exception:
        pass
    raise ValueError(f"Unrecognized asof format: {asof}")

def http_get_json(url: str, params: Dict=None, retries: int=5, backoff: float=0.5, timeout: float=12.0):
    params = params or {}
    last_exc = None
    headers = {"User-Agent": "pair-cointegration-upbit/1.5"}
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            else:
                last_exc = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_exc = e
        time.sleep(backoff * (2**i))
    raise last_exc

def _ensure_unique_sorted_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()

# ---------------- Upbit helpers ----------------

def get_krw_markets() -> List[str]:
    data = http_get_json(f"{UPBIT_BASE}/v1/market/all", params={"isDetails":"false"})
    return [d["market"] for d in data if d.get("market","").startswith("KRW-")]

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def get_top_alts_by_krw_24h(topn: int=9, exclude_btc: bool=True, exclude_stables: bool=True, sleep_sec: float=0.12) -> List[str]:
    markets = get_krw_markets()
    rows = []
    print(f"[Universe] KRW markets total={len(markets)}")
    for i, ch in enumerate(chunked(markets, 100), 1):
        tickers = http_get_json(f"{UPBIT_BASE}/v1/ticker", params={"markets":",".join(ch)})
        for t in tickers:
            market = t.get("market","")
            base = market.split("-")[1] if "-" in market else ""
            if exclude_btc and base == "BTC":
                continue
            if exclude_stables and base in STABLE_BASES:
                continue
            qv = float(t.get("acc_trade_price_24h", 0.0))
            rows.append((market, base, qv))
        print(f"  - fetched ticker chunk {i}, cumulative candidates={len(rows)}")
        time.sleep(sleep_sec)
    rows.sort(key=lambda x: x[2], reverse=True)
    top = [r[0] for r in rows[:topn]]
    print(f"[Universe] Top {topn} (ex-BTC/stables): {top}")
    return top

def _fetch_candles_series(market: str, kind: str, unit: Optional[int], target: int,
                          asof_utc: Optional[datetime], sleep_sec: float) -> pd.DataFrame:
    """
    Robust paginator (UTC 'to'):
      - Each page: pass 'to' = (oldest candle's candle_date_time_utc - 1 second) in UTC as 'YYYY-MM-DD HH:MM:SS'
      - Prevents duplicate page loops; guarantees 200->400->... growth when data exists
    """
    all_rows = []
    page = 0
    # initial 'to' — use UTC string if asof provided
    to_utc = (asof_utc.strftime("%Y-%m-%d %H:%M:%S") if asof_utc else None)
    max_pages = math.ceil(target / 200) + 10
    kind_disp = f"{kind}{unit or ''}"
    print(f"[Fetch] {market} {kind_disp} target={target} to(utc)={to_utc}")

    unique_count = 0
    last_anchor = None
    while unique_count < target and page < max_pages:
        cnt = min(200, max(1, target - unique_count))
        url = f"{UPBIT_BASE}/v1/candles/minutes/{unit}" if kind == "minutes" else f"{UPBIT_BASE}/v1/candles/days"
        params = {"market": market, "count": cnt}
        if to_utc:
            params["to"] = to_utc  # <-- UTC 'YYYY-MM-DD HH:MM:SS'
        data = http_get_json(url, params=params)
        if not data:
            print("  [Fetch] no data; break")
            break

        all_rows.extend(data)
        page += 1

        # compute uniques
        tmp = pd.DataFrame(all_rows)
        tmp["dt_utc"] = pd.to_datetime(tmp["candle_date_time_utc"], utc=True)
        tmp = tmp.set_index("dt_utc").sort_index()
        tmp = tmp[~tmp.index.duplicated(keep="last")]
        unique_count = len(tmp)

        # next 'to' anchor from oldest candle's UTC
        oldest_utc_str = data[-1]["candle_date_time_utc"]  # e.g., '2025-08-27T10:17:00'
        anchor_dt = datetime.strptime(oldest_utc_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        # fallback: if anchor identical (server returned same page), jump back harder
        if last_anchor is not None and anchor_dt == last_anchor:
            anchor_dt = anchor_dt - timedelta(minutes=(unit or 1) * cnt + 5)
        last_anchor = anchor_dt

        # set next to = anchor - 1 second (UTC)
        to_utc = (anchor_dt - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [Fetch] page {page}/{max_pages}, unique={unique_count}/{target}, next_to(utc)={to_utc}")
        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame(columns=["close","volume","timestamp"])

    df = pd.DataFrame(all_rows)
    df["dt_utc"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)
    df = df.set_index("dt_utc").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["close"] = pd.to_numeric(df["trade_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["candle_acc_trade_volume"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["close","volume","timestamp"]]

def get_klines_upbit(market: str, timeframe: str, fetch_limit: int,
                     asof_utc: Optional[datetime], sleep_sec: float) -> pd.DataFrame:
    if timeframe == "1m":
        return _fetch_candles_series(market, "minutes", 1, fetch_limit, asof_utc, sleep_sec)
    if timeframe == "1h":
        return _fetch_candles_series(market, "minutes", 60, fetch_limit, asof_utc, sleep_sec)
    if timeframe == "1d":
        return _fetch_candles_series(market, "days", None, fetch_limit, asof_utc, sleep_sec)
    raise ValueError("Unsupported timeframe")

# ---------------- Analysis ----------------

@dataclass
class PairTestResult:
    timeframe: str
    n_obs: int
    lag_k: int
    market_y: str
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

def engle_granger_with_resid_adf(x: pd.Series, y: pd.Series):
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 30:
        raise ValueError("Not enough overlapping observations for cointegration test (need >= 30).")
    tstat, pval, crit = coint(df["x"].values, df["y"].values, trend="c", maxlag=None, autolag="aic")
    X = sm.add_constant(df["x"].values)
    ols = sm.OLS(df["y"].values, X).fit()
    resid = df["y"].values - ols.predict(X)
    adf_res = adfuller(resid, autolag="AIC")
    return {
        "tstat": float(tstat),
        "pval": float(pval),
        "crit": {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
        "adf_stat": float(adf_res[0]),
        "adf_p": float(adf_res[1]),
        "adf_crit": {"1%": float(adf_res[4]["1%"]), "5%": float(adf_res[4]["5%"]), "10%": float(adf_res[4]["10%"])},
        "nused": int(adf_res[3]),
        "ols_alpha": float(ols.params[0]),
        "ols_beta": float(ols.params[1]),
    }, pd.Series(resid, index=df.index), ols

def shift_align_for_lead(y: pd.Series, k: int) -> pd.Series:
    return y.shift(-k)

def run_tests_for_market(btc_close: pd.Series, alt_close: pd.Series, timeframe: str,
                         lag_list: List[int], eval_limit: int) -> List[PairTestResult]:
    results = []
    btc_close = _ensure_unique_sorted_index(btc_close.rename("x"))
    alt_close = _ensure_unique_sorted_index(alt_close.rename(alt_close.name or "ALT"))
    for k in lag_list:
        y_lead = shift_align_for_lead(alt_close, k)
        x, y = btc_close.align(y_lead, join="inner")
        if len(x) == 0 or len(y) == 0:
            continue
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(df) >= eval_limit:
            df = df.tail(eval_limit)
        if len(df) < 30:
            continue
        r, rp = pearsonr(df["x"].values, df["y"].values)
        eg, resid, ols = engle_granger_with_resid_adf(df["x"], df["y"])
        results.append(PairTestResult(
            timeframe=timeframe, n_obs=len(df), lag_k=k, market_y=alt_close.name,
            pearson_r=float(r), pearson_p=float(rp),
            coint_t=eg["tstat"], coint_p=eg["pval"],
            coint_crit_1pct=eg["crit"]["1%"], coint_crit_5pct=eg["crit"]["5%"], coint_crit_10pct=eg["crit"]["10%"],
            adf_resid_stat=eg["adf_stat"], adf_resid_p=eg["adf_p"],
            adf_resid_crit_1pct=eg["adf_crit"]["1%"], adf_resid_crit_5pct=eg["adf_crit"]["5%"], adf_resid_crit_10pct=eg["adf_crit"]["10%"],
            ols_alpha=eg["ols_alpha"], ols_beta=eg["ols_beta"], n_used=eg["nused"],
            sample_start=df.index.min().strftime("%Y-%m-%d %H:%M:%S"),
            sample_end=df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
        ))
    return results

def fetch_all_price_series(markets: List[str], fetch_limits: Dict[str,int],
                           asof_utc: Optional[datetime], sleep_sec: float) -> Dict[str, Dict[str, pd.Series]]:
    series_map: Dict[str, Dict[str, pd.Series]] = {m: {} for m in markets}
    for tf, fetch_limit in fetch_limits.items():
        print(f"[Load] timeframe={tf}, fetch_limit={fetch_limit}")
        for m in markets:
            df = get_klines_upbit(m, tf, fetch_limit, asof_utc, sleep_sec)
            series_map[m][tf] = _ensure_unique_sorted_index(df["close"].rename(m))
            print(f"  [Load] {m} {tf} -> {len(series_map[m][tf])} bars")
    return series_map

def run_experiment(asof: Optional[str]=None, lags: List[int]=[3,5,10],
                   timeframes: Dict[str,int]={"1m":1000,"1h":500,"1d":100},
                   topn: int=9, outdir: str="./results", make_plots: bool=True, sleep_sec: float=0.12) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    asof_utc = parse_asof_utc(asof) if asof else None
    kmax = max(lags) if lags else 0
    fetch_limits = {tf: lim + kmax for tf, lim in timeframes.items()}

    top_alts = get_top_alts_by_krw_24h(topn=topn, exclude_btc=True, exclude_stables=True, sleep_sec=sleep_sec)
    btc_mkt = "KRW-BTC"
    markets = [btc_mkt] + top_alts

    series_map = fetch_all_price_series(markets, fetch_limits, asof_utc, sleep_sec)

    results: List[PairTestResult] = []
    for tf, limit in timeframes.items():
        btc_close = series_map[btc_mkt][tf].copy()
        for alt in top_alts:
            alt_close = series_map[alt][tf].copy()
            results.extend(run_tests_for_market(btc_close, alt_close, tf, lags, eval_limit=limit))

    res_df = pd.DataFrame([r.__dict__ for r in results])
    if len(res_df):
        res_df["is_cointegrated_coint_p_5pct"] = res_df["coint_p"] < 0.05
        res_df["is_cointegrated_adf_p_5pct"] = res_df["adf_resid_p"] < 0.05
        res_df["evidence_any_5pct"] = res_df["is_cointegrated_coint_p_5pct"] | res_df["is_cointegrated_adf_p_5pct"]

    stamp = ts()
    res_csv = os.path.join(outdir, f"pair_cointegration_results_{stamp}.csv")
    top_csv = os.path.join(outdir, f"top_alts_{stamp}.csv")
    res_df.to_csv(res_csv, index=False)
    pd.DataFrame({"market": top_alts}).to_csv(top_csv, index=False)

    if make_plots and len(res_df):
        import matplotlib.pyplot as plt
        for tf in sorted(res_df["timeframe"].unique()):
            sub = res_df[res_df["timeframe"]==tf].sort_values("coint_p")
            if len(sub)==0:
                continue
            row = sub.iloc[0]
            k = int(row["lag_k"])
            mkt = row["market_y"]
            btc = series_map[btc_mkt][tf]
            alt = series_map[mkt][tf]
            x, y0 = btc.align(alt, join="inner")
            merged = pd.concat([x.rename("x"), y0.shift(-k).rename("y_lead")], axis=1).dropna()
            if len(merged) >= timeframes[tf]:
                merged = merged.tail(timeframes[tf])
            if len(merged) < 30:
                continue
            plt.figure()
            plt.scatter(merged["x"].values, merged["y_lead"].values, s=6)
            plt.title(f"{tf} | {mkt} vs KRW-BTC (Y lead k={k})  | n={len(merged)}")
            plt.xlabel("KRW-BTC close"); plt.ylabel(f"{mkt} close (lead {k})")
            plot_path = os.path.join(outdir, f"scatter_{tf}_{mkt.replace('-','_')}_k{k}.png")
            plt.tight_layout(); plt.savefig(plot_path); plt.close()

    print(f"[Done] Results -> {res_csv}")
    return res_df

# ---------------- Synthetic verification ----------------

def _random_walk(n, seed=None, sigma=1.0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=sigma, size=n))

def _stationary_ar1(n, seed=None, phi=0.3, sigma=1.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=sigma, size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + e[t]
    return x

def synthetic_pair(n=1200, k=5, alpha=0.5, beta=1.7, seed=7):
    X = _random_walk(n=n, seed=seed, sigma=1.0)
    u = _stationary_ar1(n=n, seed=seed+1, phi=0.5, sigma=0.8)
    Y = np.full(n, np.nan)
    for t in range(n):
        if t-k >= 0:
            Y[t] = alpha + beta * X[t-k] + u[t]
    idx = pd.date_range("2025-01-01", periods=n, freq="T", tz="UTC")
    return pd.Series(X, index=idx, name="KRW-BTC"), pd.Series(Y, index=idx, name="KRW-ALT")

def synthetic_verify_all(lags=[3,5,10], timeframes={"1m":1000,"1h":500,"1d":100}) -> pd.DataFrame:
    out = []
    for tf, n in timeframes.items():
        for k in lags:
            Xs, Ys = synthetic_pair(n=n+10, k=k, alpha=0.3, beta=1.4, seed=42+k)
            res = run_tests_for_market(Xs.iloc[:n], Ys.iloc[:n].rename("KRW-ALT"), tf, [k], eval_limit=n)
            if res:
                r = res[0]
                out.append({"timeframe": tf, "lag_k": k, "n_obs": r.n_obs, "coint_p": r.coint_p, "adf_resid_p": r.adf_resid_p})
    return pd.DataFrame(out)

# ---------------- CLI ----------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Upbit cointegration (UTC 'to' pagination, strict windows)")
    p.add_argument("--run", action="store_true")
    p.add_argument("--asof", type=str, default=None, help="UTC end time, e.g. '2025-08-27 07:00:00'")
    p.add_argument("--lags", type=int, nargs="+", default=[3,5,10])
    p.add_argument("--outdir", type=str, default="./results")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--topn", type=int, default=9)
    p.add_argument("--sleep-sec", type=float, default=0.12)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    if args.verify or not args.run:
        print("[Verify] synthetic pipeline...")
        ver = synthetic_verify_all(lags=args.lags)
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, f"synthetic_verification_{ts()}.csv")
        ver.to_csv(out, index=False)
        print("->", out)
        print(ver.head())
        if not args.run:
            return

    print(f"[Run] asof(UTC)={args.asof} lags={args.lags} sleep={args.sleep_sec}")
    df = run_experiment(
        asof=args.asof, lags=args.lags, outdir=args.outdir,
        make_plots=(not args.no_plots), topn=args.topn, sleep_sec=args.sleep_sec
    )
    out = os.path.join(args.outdir, f"pair_cointegration_results_{ts()}.csv")
    df.to_csv(out, index=False)
    print("->", out)

if __name__ == "__main__":
    main()
