# Cointegration Results Columns – Quick Reference
*Generated: 2025-08-27 07:15:08 UTC*

각 컬럼의 **정의 / 사용처 / 해석 포인트**를 간략히 정리했습니다.

---

## 기본 메타
- **timeframe**: 사용 캔들 주기. `1m`(1분), `1h`(1시간), `1d`(1일).
- **n_obs**: 최종 분석에 **실제로 사용된 관측치 개수**. (리드·정렬·결측 제거 후 마지막 N개 슬라이스 결과)
- **lag_k**: 리드 스텝 k. `Y(t+k)`를 만들기 위해 `Y.shift(-k)` 적용.
- **market_y**: 대상 알트코인(또는 종목) 심볼(예: `KRW-ETH`).

## 상관
- **pearson_r**: 피어슨 상관계수(−1~1). 동시(정렬된) 시점의 **선형 동행 정도**.
- **pearson_p**: 상관이 0이라는 귀무가설의 **p-value**. 작을수록 상관 유의.

## Engle–Granger(공적분) – 1단계
- **coint_t**: EG 공적분 테스트 통계량. **더 작을수록**(보통 음수 방향) 공적분 증거 강함.
- **coint_p**: EG 공적분 테스트의 **p-value**. 작을수록 공적분 유의.
- **coint_crit_1pct / 5pct / 10pct**: 해당 유의수준의 임계값. `coint_t`가 임계값보다 **작으면** 기각.

## 잔차 ADF(공적분) – 2단계
- **adf_resid_stat**: EG 1단계 회귀 잔차 \(e_t\)에 대한 **ADF 통계량**. **더 작을수록**(음수) 정상성 증거 ↑.
- **adf_resid_p**: 잔차 ADF의 **p-value**. 작을수록 잔차가 정상적(=공적분과 일관).
- **adf_resid_crit_1pct / 5pct / 10pct**: ADF 임계값들.
- **n_used**: ADF 회귀에서 **실제 사용된 유효 표본 수**(시차 포함 후).

## 회귀 계수(헤지 정보)
- **ols_alpha**: EG 1단계 OLS 절편 \(\alpha\).
- **ols_beta**: EG 1단계 OLS 기울기 \(\beta\) (**헤지비율**). 스프레드 \(Y-\alpha-\beta X\) 구성에 사용.

## 샘플 구간
- **sample_start / sample_end**: 해당 페어·라그·주기의 **사용 구간 시간 범위**(UTC).

## 유의성 플래그(단순 기준, 5%)
- **is_cointegrated_coint_p_5pct** (= **is_coint_5**): `coint_p < 0.05`이면 `True`.
- **is_cointegrated_adf_p_5pct** (= **is_adf_5**): `adf_resid_p < 0.05`이면 `True`.
- **evidence_any_5pct** (= **either_5**): 위 둘 중 **하나라도** 참이면 `True`.
- **both_5**: 두 기준을 **동시에** 만족(`True` ∧ `True`) → **강한 공적분 후보**.

## 점수 및 다중검정(FDR)
- **score**: 공적분 강도 종합 점수. `score = -log10(coint_p) + -log10(adf_resid_p)` → **클수록 강함**.
- **bh_coint**: (같은 `timeframe`, `lag_k` 그룹 내) `coint_p`에 **Benjamini–Hochberg FDR(q=0.05)** 적용 결과.
- **bh_adf**: 동일 그룹에서 `adf_resid_p`의 **BH-FDR(q=0.05)** 결과.
- **bh_both**: `bh_coint` **AND** `bh_adf` 둘 다 `True` → **다중검정까지 통과한 견고 후보**.

---

### 빠른 해석 팁
- **선정 1순위**: `both_5=True`이면서 `bh_both=True`, `score`가 큰 페어.
- **주의**: `pearson_p`는 **단기 상관** 지표로, 공적분(장기균형) 판단의 보조 지표로 사용.
