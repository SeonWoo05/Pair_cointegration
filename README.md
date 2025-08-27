# Pair Cointegration Code – Summary Report

본 문서는 프로젝트의 주요 스크립트 3개에 대한 간략 설명과 사용법, 공통 로직 및 차이점을 요약합니다.

## pair_cointegration_binance.py
> pair_cointegration_binance.py

**기본 상수:**
- `BINANCE_BASE` = "https://api.binance.com"

**주요 함수 목록 (16):**
`ts`, `to_ms`, `parse_asof`, `http_get_json`, `get_top_alts_by_quote_volume`, `get_klines`, `engle_granger_with_resid_adf`, `shift_align_for_lead`, `run_tests_for_symbol`, `fetch_all_price_series`, `run_experiment`, `_random_walk`, `_stationary_ar1`, `synthetic_pair`, `synthetic_verify_all`, `main``

**CLI 옵션 (argparse):**
- `--run`  → action="store_true", help="Run the full experiment (requires internet
- `--asof`  → type=str, default=None, help="End time (UTC
- `--quote`  → type=str, default="USDT", help="Quote asset (default USDT
- `--lags`  → type=int, nargs="+", default=[3,5,10], help="Lags k to test."
- `--outdir`  → type=str, default="./results", help="Output directory."
- `--no-plots`  → action="store_true", help="Disable scatter plots."
- `--topn`  → type=int, default=9, help="Number of altcoins to test (default 9
- `--verify`  → action="store_true", help="Run synthetic verification only (no internet

## pair_cointegration_upbit.py
> pair_cointegration_upbit_converted_v3.py

**기본 상수:**
- `UPBIT_BASE` = "https://api.upbit.com"

**주요 함수 목록 (19):**
`ts`, `parse_asof_utc`, `http_get_json`, `_ensure_unique_sorted_index`, `get_krw_markets`, `chunked`, `get_top_alts_by_krw_24h`, `_fetch_candles_series`, `get_klines_upbit`, `engle_granger_with_resid_adf`, `shift_align_for_lead`, `run_tests_for_market`, `fetch_all_price_series`, `run_experiment`, `_random_walk`, `_stationary_ar1`, `synthetic_pair`, `synthetic_verify_all`` …`

**CLI 옵션 (argparse):**
- `--run`  → action="store_true"
- `--asof`  → type=str, default=None, help="UTC end time, e.g. '2025-08-27 07:00:00'"
- `--lags`  → type=int, nargs="+", default=[3,5,10]
- `--outdir`  → type=str, default="./results"
- `--no-plots`  → action="store_true"
- `--topn`  → type=int, default=9
- `--sleep-sec`  → type=float, default=0.12
- `--verify`  → action="store_true"

## upbit_cointegration_report_builder.py
> upbit_cointegration_report_builder.py

**주요 함수 목록 (8):**
`_now_utc_str`, `_bh_fdr`, `_ensure_name_col`, `_safe_log10_p`, `_make_histogram`, `build_report_from_csv`, `_fmt_pair_row`, `md_to_html_simple``

**CLI 옵션 (argparse):**
- `--results_csv`  → required=True, help="Path to results CSV (from pair_cointegration_upbit.py
- `--top_alts_csv`  → default=None, help="Optional path to top_alts CSV."
- `--outdir`  → default="./report", help="Output directory."
- `--alpha`  → type=float, default=0.05, help="Significance level for tests."
- `--fdr_q`  → type=float, default=0.05, help="FDR q for Benjamini-Hochberg."
- `--no-scatter`  → action="store_true", help="Do not include scatter images even if present."

---
## 공통 파이프라인 (요약)
1) **유니버스 선택**: 상위 거래대금(24h) 알트 9개 (BTC/스테이블 제외).
2) **캔들 수집**: 1m=1000, 1h=500, 1d=100 (리드 k 보정 위해 `+max(lags)` 만큼 여유 수집 후 마지막 N개 슬라이스).
3) **정렬/리드**: `Y_lead = Y.shift(-k)`, BTC와 시계열 **교집합 정렬** 및 결측 제거.
4) **검정**: Pearson r/p, Engle–Granger cointegration(`statsmodels.coint`), 잔차 ADF(`adfuller`).
5) **판정 플래그**: `is_cointegrated_coint_p_5pct`, `is_cointegrated_adf_p_5pct`, `evidence_any_5pct`.
6) **출력**: 결과 CSV, 상위 알트 CSV, (옵션) 산점도 PNG 및 보고서 도표.

### 차이점
- **Binance 스크립트**: 심볼 체계(예: `BTCUSDT`), `quoteVolume` 기준 랭킹, `/api/v3/klines`.
- **Upbit 스크립트**: 마켓 체계(예: `KRW-BTC`), `acc_trade_price_24h` 기준 랭킹, `/v1/candles` 계열 API. 
  - Upbit는 요청당 최대 200개 → **UTC `to` 기반 페이지네이션**으로 다중 페이지 수집을 보장해야 함.

## 통계 검정(정확 설명)
- **EG 1단계(OLS)**: `Y_t = α + β X_t + e_t` → β는 **헤지비율**.
- **EG 2단계(잔차 ADF)**: `Δe_t = μ + ρ e_{t-1} + Σ φ_i Δe_{t-i} + ε_t`  
  귀무: 단위근(비정상). `adf_resid_p`가 작을수록 정상성(=공적분) 증거 ↑.
- **Engle–Granger coint p-value (`coint_p`)**: 두 시계열이 공적분인지 직접 검정. 작을수록 유의.
- **Lead(k)**: (X(t), Y(t+k)) 관계를 검정. 연구용 패턴 파악에 유효하지만, 백테스트는 **롤링 추정**으로 미래정보 누출 방지 필요.

## 주요 입출력
- **입력 파라미터**: `--run/--verify`, `--asof`, `--lags`, `--outdir`, `--topn`, (Upbit) `--sleep-sec`, (Binance) `--quote`.
- **출력 파일**: `pair_cointegration_results_<ts>.csv`, `top_alts_<ts>.csv`, (옵션) `scatter_<tf>_<symbol>_k<k>.png`.

## 주의/권고
- Upbit 1분봉은 **페이지네이션** 실패 시 200개에서 멈춤 → UTC `to = oldest_utc - 1s` 로 강제 과거이동.
- **교집합 부족**(신규상장·빈봉) 시 `n_obs`가 목표(1000/500/100)보다 작아질 수 있음.
- 다중검정 보정(FDR)을 사용하는 보고서 빌더(`upbit_cointegration_report_builder.py`)로 **견고 후보**만 선별 권장.
- 전략화 단계: 롤링 재추정, 스프레드 z-score 엔트리/청산, 거래비용/슬리피지 반영 백테스트 필수.

## 실행 예시
```bash
# 합성 검증
python pair_cointegration_upbit.py --verify --outdir ./results
# 실데이터(업비트)
python pair_cointegration_upbit.py --run --outdir ./results --sleep-sec 0.2
# 실데이터(바이낸스)
python pair_cointegration_binance.py --run --outdir ./results
```
