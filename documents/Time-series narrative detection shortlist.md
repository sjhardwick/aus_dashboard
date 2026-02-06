**Goal:** surface *candidate economic stories* (trend, regime change, momentum, shocks, suppression) that would be unlikely under pure noise, without claiming causal proof.

---

## 1. Rolling linear trend (slope + sign stability)

**Run**
- OLS trend over rolling windows (e.g. 3–5 years)
- Track slope, sign, and standardized t-stat

**Flags**
- Persistent positive/negative slope
- Acceleration or deceleration
- Long flat periods (slope ≈ 0)

**Story detected**
- Secular growth or decline
- Trend breaks
- “Lost decade” dynamics

---

## 2. ADF vs KPSS (joint interpretation)

**Run**
- ADF (null: unit root)
- KPSS (null: stationarity)
- Compare across subsamples and transformations

**Flags**
- ADF fails to reject + KPSS rejects → drift-like behavior
- Regime-dependent persistence

**Story detected**
- Accumulating pressure
- Mean reversion vs structural drift

---

## 3. Bai–Perron multiple structural breaks (mean or trend)

**Run**
- Multiple break test on levels and/or trend

**Flags**
- Statistically distinct regimes
- Breaks clustered around specific dates

**Story detected**
- Policy or institutional regime change
- Measurement redefinition
- Technology or market structure shifts

---

## 4. CUSUM / recursive residual stability

**Run**
- Recursive estimation
- CUSUM and/or CUSUMSQ tests

**Flags**
- Gradual instability before a clean break
- Persistent parameter drift

**Story detected**
- Institutional erosion
- Credibility decay
- Slow-moving political pressure

---

## 5. Rolling variance / volatility breaks

**Run**
- Rolling standard deviation
- Variance break tests (e.g. ICSS, Bai–Perron on variance)

**Flags**
- Sharp or sustained volatility changes
- Volatility breaks without mean breaks

**Story detected**
- Risk repricing
- Financialisation or repression
- Policy credibility gains/losses

---

## 6. Last-observation extremeness (endpoint diagnostics)

**Run**
- Z-score of latest observation relative to:
  - recent window (e.g. 3–5 years)
  - full sample

**Flags**
- Tail observations
- Endpoint flips trend sign

**Story detected**
- Shocks
- Turning points
- Crisis or recovery onset

---

## 7. Growth vs base-effect decomposition

**Run**
- Compare:
  - YoY growth
  - QoQ annualised
  - Counterfactual growth from fixed base

**Flags**
- Large growth driven by low base
- Strong YoY masking weak momentum

**Story detected**
- Mechanical rebounds
- Fake “V-shaped” recoveries

---

## 8. Rolling AR(1) persistence

**Run**
- Estimate AR(1) coefficient over rolling windows

**Flags**
- Rising persistence
- Sudden collapse in autocorrelation

**Story detected**
- Momentum emergence or breakdown
- Policy transmission changes
- Narrative entrenchment

---

## 9. Runs test / sign clustering

**Run**
- Runs test on first differences or growth rates

**Flags**
- Too many consecutive moves in same direction
- Suppressed reversals

**Story detected**
- Directional pressure
- Administrative smoothing
- Hidden constraints

---

## 10. Rolling correlations with peers (multivariate)

**Run**
- Rolling correlations with related series, sectors, or regions

**Flags**
- Sudden decoupling or convergence
- Correlation regime shifts

**Story detected**
- Fragmentation vs integration
- Contagion
- Breakdown of historical relationships

---

## How to use this

- Treat these as **tripwires**, not proof.
- If several tests activate in the same window, the probability of a meaningful economic narrative rises sharply.
- Only *after* this step should institutional or economic context be layered in.

---

## Rough mapping: tests → stories

| Story type | Tests |
|----------|-------|
| Secular trend | 1, 2 |
| Regime change | 3, 4 |
| Shock / crisis | 5, 6 |
| False recovery | 7 |
| Momentum | 8, 9 |
| Suppression | 5, 9 |
| Fragmentation | 10 |
| Credibility loss | 4, 5 |
| Anticipation effects | 4, 8 |
| Measurement change | 3 + 5 mismatch |