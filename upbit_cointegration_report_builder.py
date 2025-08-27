
"""
upbit_cointegration_report_builder.py

Builds a self-contained Markdown & HTML report from cointegration test results.

Inputs:
- results_csv: CSV from pair_cointegration_upbit.py (or a compatible schema with columns below)
- Optional: top_alts_csv for listing top markets (not required)
- outdir: output directory
- alpha: significance level (default 0.05)
- fdr_q: Benjamini-Hochberg FDR level (default 0.05)

Outputs:
- <outdir>/cointegration_report.md
- <outdir>/cointegration_report.html
- <outdir>/summary_top_overall.csv
- <outdir>/summary_top_by_timeframe.csv
- <outdir>/summary_counts_by_tf_k.csv
- Figures (*.png): per-timeframe histograms of -log10(p) for coint/adf, plus (optional) scatter images if exist.

Usage (as a library):
    from upbit_cointegration_report_builder import build_report_from_csv
    build_report_from_csv(results_csv, top_alts_csv=None, outdir="./report", alpha=0.05, fdr_q=0.05)

Author: ChatGPT (GPT-5 Pro)
"""
import os, io, math, json
from typing import Optional, List
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLS = [
    "timeframe","n_obs","lag_k","pearson_r","pearson_p","coint_t","coint_p",
    "adf_resid_stat","adf_resid_p","ols_alpha","ols_beta","sample_start","sample_end"
]
# One of these should exist to name the Y market
NAME_COL_CANDIDATES = ["market_y","symbol_y","market","symbol"]

def _now_utc_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def _bh_fdr(pvals: np.ndarray, q: float=0.05) -> np.ndarray:
    """
    Benjamini-Hochberg step-up FDR control. Return boolean array of discoveries.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n+1) / n)
    passed = ranked <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        keep[:kmax+1] = True
    result = np.zeros(n, dtype=bool)
    result[order] = keep
    return result

def _ensure_name_col(df: pd.DataFrame) -> str:
    for c in NAME_COL_CANDIDATES:
        if c in df.columns:
            return c
    # If nothing found, create a placeholder
    df["market_y"] = [f"ALT_{i}" for i in range(len(df))]
    return "market_y"

def _safe_log10_p(p):
    return -np.log10(np.clip(p, 1e-300, 1.0))

def _make_histogram(series: pd.Series, title: str, out_path: str):
    plt.figure()
    vals = series.dropna().values
    if len(vals) == 0:
        # Create an empty plot with a note
        plt.title(title + " (no data)")
        plt.savefig(out_path)
        plt.close()
        return
    plt.hist(vals, bins=30)
    plt.title(title)
    plt.xlabel("-log10(p)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def build_report_from_csv(results_csv: str, top_alts_csv: Optional[str]=None, outdir: str="./report",
                          alpha: float=0.05, fdr_q: float=0.05, include_scatter: bool=True):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(results_csv)

    # Validate / adapt schema
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")
    name_col = _ensure_name_col(df)

    # Compute flags and scores
    df["is_coint_5"] = df["coint_p"] < alpha
    df["is_adf_5"] = df["adf_resid_p"] < alpha
    df["both_5"] = df["is_coint_5"] & df["is_adf_5"]
    df["either_5"] = df["is_coint_5"] | df["is_adf_5"]
    df["score"] = _safe_log10_p(df["coint_p"]) + _safe_log10_p(df["adf_resid_p"])

    # FDR per (timeframe, lag_k)
    df["bh_coint"] = False
    df["bh_adf"] = False
    for (tf, k), sub_idx in df.groupby(["timeframe","lag_k"]).groups.items():
        idx = list(sub_idx)
        c_keep = _bh_fdr(df.loc[idx, "coint_p"].values, q=fdr_q)
        a_keep = _bh_fdr(df.loc[idx, "adf_resid_p"].values, q=fdr_q)
        df.loc[idx, "bh_coint"] = c_keep
        df.loc[idx, "bh_adf"] = a_keep
    df["bh_both"] = df["bh_coint"] & df["bh_adf"]

    # Rankings
    # Overall (strict: both_5) then by score
    strict = df[df["both_5"]].copy()
    strict_sorted = strict.sort_values(["score","n_obs"], ascending=[False, False])
    top_overall = strict_sorted.head(20)
    top_overall.to_csv(os.path.join(outdir, "summary_top_overall.csv"), index=False)

    # Top by timeframe (strict)
    top_by_tf_rows = []
    for tf, g in strict.groupby("timeframe"):
        g_sorted = g.sort_values(["score","n_obs"], ascending=[False, False]).head(10)
        top_by_tf_rows.append(g_sorted)
    top_by_tf = pd.concat(top_by_tf_rows) if len(top_by_tf_rows) else pd.DataFrame(columns=df.columns)
    top_by_tf.to_csv(os.path.join(outdir, "summary_top_by_timeframe.csv"), index=False)

    # Counts by timeframe/lag
    cnt_rows = []
    for (tf, k), g in df.groupby(["timeframe","lag_k"]):
        cnt_rows.append({
            "timeframe": tf,
            "lag_k": k,
            "n_pairs": len(g),
            "n_coint_5": int(g["is_coint_5"].sum()),
            "n_adf_5": int(g["is_adf_5"].sum()),
            "n_both_5": int(g["both_5"].sum()),
            "n_bh_both": int(g["bh_both"].sum()),
            "avg_n_obs": float(g["n_obs"].mean())
        })
    counts_df = pd.DataFrame(cnt_rows).sort_values(["timeframe","lag_k"])
    counts_df.to_csv(os.path.join(outdir, "summary_counts_by_tf_k.csv"), index=False)

    # Figures: histograms of -log10(p) for each timeframe
    for tf, g in df.groupby("timeframe"):
        _make_histogram(_safe_log10_p(g["coint_p"]), title=f"{tf}  -log10(coint_p) distribution",
                        out_path=os.path.join(outdir, f"hist_{tf}_coint_p.png"))
        _make_histogram(_safe_log10_p(g["adf_resid_p"]), title=f"{tf}  -log10(adf_resid_p) distribution",
                        out_path=os.path.join(outdir, f"hist_{tf}_adf_resid_p.png"))

    # Compose Markdown report
    now = _now_utc_str()
    title = "# 업비트 공적분 탐색 보고서\n"
    subtitle = f"*생성 시각: {now}*\n\n"

    # Metrics explanation (Korean)
    expl = """
## 1) 지표 설명

- **피어슨 상관계수 (r, p)**: 동시시점 상관관계의 선형강도와 유의성입니다. *공적분*은 단기 상관과는 다른 개념이므로, r이 낮아도 공적분일 수 있습니다.
- **Engle–Granger (EG) 공적분 검정**: 두 비정상 시계열 X, Y가 특정 선형결합으로 정상성을 가지는지(=장기균형관계 존재)를 검정합니다. 여기서는 `coint_p`가 작을수록 공적분 증거가 강합니다.
- **잔차 ADF (단위근) 검정**: 회귀 \(Y_t = \\alpha + \\beta X_t + e_t\)의 잔차 \(e_t\)가 정상성을 가지는지를 검정합니다. `adf_resid_p`가 작을수록 정상성(=공적분 일관성) 증거가 강합니다.
- **헤지비율(OLS β)**: EG 1단계 회귀에서의 기울기 추정치로, 스프레드 구성 \(e_t = Y_t - \\alpha - \\beta X_t\)에 쓰입니다.
- **리드 테스트**: 본 연구는 (X(t), Y(t+k))를 검정합니다. 즉 Y를 k스텝 **리드**하여 미래값이 X의 현재와 장기관계를 갖는지 확인합니다. 전략화 단계에서는 미래정보 누출을 피하기 위해 롤링/아웃오브샘플 검증이 필수입니다.
- **판정 기준**: 기본 5% 유의수준에서 `coint_p<0.05` AND `adf_resid_p<0.05`를 **엄격한(Strict)** 기준으로, 둘 중 하나라도 5%면 **완화(Loose)** 증거로 표기했습니다. 또한 (timeframe, k) 그룹 내부에서 **Benjamini–Hochberg FDR(0.05)**로 다중검정을 보정했습니다.
"""

    # Methods and data window
    windows = df.groupby("timeframe")["n_obs"].agg(["min","max","mean"]).reset_index()
    windows_md = windows.to_markdown(index=False)

    methods = f"""
## 2) 데이터 및 방법

- 거래소: **Upbit KRW 마켓** (X=KRW-BTC, Y=상위 9개 알트)
- 캔들: 1분봉 1000개, 1시간봉 500개, 1일봉 100개 (요청 시점 기준 과거)
- 리드(선행) 라그: k ∈ {{3, 5, 10}}
- 공적분 검정: EG + 잔차 ADF, 유의수준 α=0.05, FDR q=0.05
- 실험 대상 페어 수: {len(df)}  (결측 제거 및 표본<30 제외 후 기준)
- 사용 표본 개수 통계(관측치 n_obs):
{windows_md}
"""

    # Findings: top pairs
    def _fmt_pair_row(row):
        return f"- **{row[name_col]}** | tf={row['timeframe']}, k={int(row['lag_k'])}, n={int(row['n_obs'])},  " \
               f"score={row['score']:.2f}, coint_p={row['coint_p']:.2e}, adf_p={row['adf_resid_p']:.2e}, β={row['ols_beta']:.10f}"

    findings_lines = []
    findings_lines.append("## 3) 결과 요약\n")
    if len(strict_sorted):
        best_row = strict_sorted.iloc[0]
        findings_lines.append("### 3.1 전체 Top 페어 (엄격 기준; both_5)\n")
        findings_lines.append(_fmt_pair_row(best_row) + "\n")
    else:
        findings_lines.append("엄격 기준(both_5)으로 유의한 페어가 없습니다.\n")

    findings_lines.append("### 3.2 타임프레임별 Top 3 (엄격 기준)\n")
    for tf, g in strict_sorted.groupby("timeframe"):
        sub = g.head(3)
        if len(sub)==0:
            findings_lines.append(f"- {tf}: (해당 없음)\n")
        else:
            findings_lines.append(f"- {tf}:\n")
            for _, row in sub.iterrows():
                findings_lines.append(_fmt_pair_row(row))

    # Counts table
    counts_md = counts_df.to_markdown(index=False)

    findings_lines.append("\n### 3.3 (timeframe, k)별 유의 페어 수\n")
    findings_lines.append(counts_md + "\n")

    # Interpretation guidance
    guidance = """
## 4) 해석 가이드 & 추천

- **가장 공적분에 잘 맞는 페어**는 (a) `both_5`를 만족하고, (b) `score=-log10(coint_p)-log10(adf_p)`가 큰 순입니다.
- 동일 (timeframe, k) 내 다중검정 보정(BH-FDR 5%)을 통과한 `bh_both` 페어는 보다 **견고한 후보**로 간주할 수 있습니다.
- 실전 전략화 시:
  1) 롤링 윈도우로 β/α 재추정 및 스프레드 정상성 재검정(워크포워드)  
  2) 스프레드 z-score 엔트리/청산 + 거래비용/슬리피지 반영 백테스트  
  3) 라그(k) 및 주기별 민감도 분석, 안정성 높은 조합 선호
"""

    # Attach figures list
    figs = []
    for tf in sorted(df["timeframe"].unique()):
        figs.append(f"hist_{tf}_coint_p.png")
        figs.append(f"hist_{tf}_adf_resid_p.png")
    figs_exist = [f for f in figs if os.path.exists(os.path.join(outdir, f))]

    fig_md_lines = []
    if figs_exist:
        fig_md_lines.append("\n## 5) p-value 분포 도표\n")
        for f in figs_exist:
            fig_md_lines.append(f"![{f}]({f})")

    # Optional: list scatter plots if present in outdir
    if include_scatter:
        scatters = [fn for fn in os.listdir(outdir) if fn.startswith("scatter_") and fn.endswith(".png")]
        if scatters:
            fig_md_lines.append("\n## 6) 산점도(상위 페어 예시)\n")
            for s in sorted(scatters)[:12]:
                fig_md_lines.append(f"![{s}]({s})")

    # Build Markdown content
    md_parts = [
        title, subtitle,
        expl,
        methods,
        "\n".join(findings_lines),
        guidance,
        "\n".join(fig_md_lines)
    ]
    md_content = "\n\n".join(md_parts)
    md_path = os.path.join(outdir, "cointegration_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Simple HTML wrapper (no external deps)
    def md_to_html_simple(md_text: str) -> str:
        # Very naive markdown-to-html for headers/lists/images/code; keeps it simple
        html = md_text
        # headers
        for i in range(6,0,-1):
            html = html.replace("\n" + "#"*i + " ", f"\n<h{i}>") \
                       .replace("\n" + "#"*i + "\n", f"</h{i}>\n")
        # bold/italics
        html = html.replace("**", "<b>").replace("*", "<i>")
        # images
        html = html.replace("![", '<img alt="').replace("](", '" src="').replace(")", '">')
        # lists
        html = html.replace("\n- ", "\n<li>")
        html = html.replace("\n", "<br/>\n")
        # wrap
        return f"<html><head><meta charset='utf-8'><title>Upbit Cointegration Report</title></head><body>{html}</body></html>"

    html_content = md_to_html_simple(md_content)
    html_path = os.path.join(outdir, "cointegration_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "md_path": md_path,
        "html_path": html_path,
        "top_overall_csv": os.path.join(outdir, "summary_top_overall.csv"),
        "top_by_tf_csv": os.path.join(outdir, "summary_top_by_timeframe.csv"),
        "counts_csv": os.path.join(outdir, "summary_counts_by_tf_k.csv"),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build a Markdown & HTML report from cointegration results CSV.")
    p.add_argument("--results_csv", required=True, help="Path to results CSV (from pair_cointegration_upbit.py).")
    p.add_argument("--top_alts_csv", default=None, help="Optional path to top_alts CSV.")
    p.add_argument("--outdir", default="./report", help="Output directory.")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level for tests.")
    p.add_argument("--fdr_q", type=float, default=0.05, help="FDR q for Benjamini-Hochberg.")
    p.add_argument("--no-scatter", action="store_true", help="Do not include scatter images even if present.")
    args = p.parse_args()
    res = build_report_from_csv(results_csv=args.results_csv,
                                top_alts_csv=args.top_alts_csv,
                                outdir=args.outdir,
                                alpha=args.alpha,
                                fdr_q=args.fdr_q,
                                include_scatter=(not args.no_scatter))
    print(json.dumps(res, indent=2, ensure_ascii=False))
