"""
Portfolio optimizer: find weights over S&P 500 constituents that achieve
~2% tracking error and ~1% active alpha vs. SPY.

Approach
--------
1. Screen to stocks with positive active FF5 alpha (alpha_i > alpha_SPY).
   ~250 stocks pass; this preserves idiosyncratic diversification needed
   to reach a 2% TE floor, while excluding negative-alpha detractors.
2. Compute an analytical max-IR warm start via the Woodbury matrix identity
   (Σ⁻¹ α in O(NK + K³)), then project to long-only / bounded simplex.
3. Run SLSQP from that warm start for two strategies:
     a. Max-alpha  : maximize active alpha | TE <= 2%
     b. Min-TE     : minimize TE           | active alpha >= 1%

NOTE: alphas are in-sample (2-yr historical). They capture the fitted period,
not forward-looking expected returns. Use as an illustrative framework.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from risk_model import load_returns, fetch_ff5, estimate_loadings

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_TE    = 0.02    # 2% annualised
TARGET_ALPHA = 0.01    # 1% annualised active alpha vs SPY
MAX_WEIGHT   = 0.05    # 5% per stock
FACTOR_COLS  = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


# ── TE and analytic gradient ──────────────────────────────────────────────────

def te_and_grad(w, B, b_spy, sigma_F, d):
    delta_b = B.T @ w - b_spy
    te2     = delta_b @ sigma_F @ delta_b + np.dot(w * w, d)
    te      = np.sqrt(max(te2, 1e-20))
    grad    = (B @ (sigma_F @ delta_b) + w * d) / te
    return te, grad


# ── Analytical max-IR warm start (Woodbury) ───────────────────────────────────

def max_ir_warm_start(active_alpha, B, sigma_F, d, max_w):
    """
    Closed-form maximum-IR weights via the Woodbury identity:
        Σ⁻¹ α = D⁻¹ α − D⁻¹ B (Σ_F⁻¹ + B' D⁻¹ B)⁻¹ B' D⁻¹ α
    Projects result to [0, max_w] long-only simplex.
    """
    d_inv    = 1.0 / np.maximum(d, 1e-10)
    a        = d_inv * active_alpha
    BtDinv_a = B.T @ a                        # (K,)
    M        = np.linalg.inv(sigma_F) + (B.T * d_inv) @ B  # (K, K)
    z        = np.linalg.solve(M, BtDinv_a)   # (K,)
    w_raw    = a - d_inv * (B @ z)            # Σ⁻¹ α, shape (N,)

    # Project to positive simplex with box constraint
    w = np.maximum(w_raw, 0.0)
    if w.sum() < 1e-12:
        return np.ones(len(w)) / len(w)
    w = np.minimum(w / w.sum(), max_w)
    w /= w.sum()
    return w


# ── Build model matrices ──────────────────────────────────────────────────────

def build_inputs(betas, idio_var, returns, factors):
    stock_tickers = [t for t in betas.index if t != "SPY"]
    spy_alpha = float(betas.loc["SPY", "alpha"]) * 252
    spy_betas = betas.loc["SPY", FACTOR_COLS].values.astype(float)

    # Screen: keep only stocks with positive active alpha
    all_alphas    = betas.loc[stock_tickers, "alpha"] * 252
    active_alphas = all_alphas - spy_alpha
    pos_mask      = active_alphas > 0
    tickers       = active_alphas[pos_mask].sort_values(ascending=False).index.tolist()

    print(f"  Stocks with positive active alpha: {len(tickers)} "
          f"(range: {active_alphas[pos_mask].min()*100:.1f}% – "
          f"{active_alphas[pos_mask].max()*100:.1f}%)")

    B         = betas.loc[tickers, FACTOR_COLS].values.astype(float)
    d         = idio_var.reindex(tickers).fillna(idio_var.median()).values
    alpha_vec = betas.loc[tickers, "alpha"].values * 252
    act_alpha = alpha_vec - spy_alpha

    common  = returns.index.intersection(factors.index)
    sigma_F = factors.loc[common, FACTOR_COLS].cov().values * 252

    return tickers, B, spy_betas, sigma_F, d, alpha_vec, act_alpha, spy_alpha


# ── Optimisation helpers ──────────────────────────────────────────────────────

def run_opt(objective, jac, constraints, bounds, w0, label):
    print(f"\n  Running: {label} ...")
    res = minimize(
        objective, w0, jac=jac, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9, "disp": False},
    )
    ok = res.success or "Positive directional" in res.message
    if not ok:
        print(f"  [warn] {res.message}")
    return res


# ── Report ────────────────────────────────────────────────────────────────────

def report(label, w, tickers, B, b_spy, sigma_F, d, act_alpha, betas):
    te, _     = te_and_grad(w, B, b_spy, sigma_F, d)
    delta_b   = B.T @ w - b_spy
    f_te2     = delta_b @ sigma_F @ delta_b
    i_te2     = np.dot(w * w, d)
    port_aa   = act_alpha @ w      # active alpha

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Tracking error (annualised) : {te*100:.2f}%")
    print(f"    Factor TE                 : {np.sqrt(max(f_te2,0))*100:.2f}%")
    print(f"    Idiosyncratic TE          : {np.sqrt(max(i_te2,0))*100:.2f}%")
    print(f"  Active alpha vs SPY         : {port_aa*100:.2f}%")
    print(f"  Information ratio           : {port_aa/te:.2f}")
    print(f"  Stocks held (weight > 0.1%) : {(w > 0.001).sum()}")
    print(f"  Max single-stock weight     : {w.max()*100:.2f}%")

    top10 = pd.DataFrame({
        "ticker":      tickers,
        "weight_%":    w * 100,
        "act_alph_%":  act_alpha * 100,
        "mkt_beta":    betas.loc[tickers, "Mkt-RF"].values,
    }).sort_values("weight_%", ascending=False).head(10)
    print(f"\n  Top 10 holdings:")
    print(top10.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n  Active factor exposures (portfolio - SPY):")
    for i, f in enumerate(FACTOR_COLS):
        print(f"    {f:8s}  {delta_b[i]:+.4f}")

    return {
        "strategy":       label,
        "te_%":           round(te * 100, 4),
        "factor_te_%":    round(np.sqrt(max(f_te2, 0)) * 100, 4),
        "idio_te_%":      round(np.sqrt(max(i_te2, 0)) * 100, 4),
        "active_alpha_%": round(port_aa * 100, 4),
        "info_ratio":     round(port_aa / te, 4),
        "n_stocks":       int((w > 0.001).sum()),
        "max_weight_%":   round(w.max() * 100, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Portfolio Optimizer — 2% TE / 1% Alpha vs SPY")
    print("=" * 60)

    print("\nLoading data and estimating factor loadings ...")
    returns = load_returns()
    factors = fetch_ff5()
    betas, idio_var = estimate_loadings(returns, factors)
    print(f"  Full universe: {len(betas)-1} stocks + SPY")
    print(f"  SPY FF5 alpha: {betas.loc['SPY','alpha']*252*100:.2f}%/yr")

    tickers, B, b_spy, sigma_F, d, _, act_alpha, _ = build_inputs(
        betas, idio_var, returns, factors
    )
    N      = len(tickers)
    bounds = [(0.0, MAX_WEIGHT)] * N

    # Analytical warm start
    print(f"\nComputing analytical max-IR warm start (Woodbury) ...")
    w_warm = max_ir_warm_start(act_alpha, B, sigma_F, d, MAX_WEIGHT)
    te_warm, _ = te_and_grad(w_warm, B, b_spy, sigma_F, d)
    aa_warm    = act_alpha @ w_warm
    print(f"  Warm-start TE: {te_warm*100:.2f}%  |  active alpha: {aa_warm*100:.2f}%")

    eq_con = {"type": "eq",
              "fun": lambda w: w.sum() - 1,
              "jac": lambda _: np.ones(N)}

    # ── Strategy 1: maximise active alpha | TE <= 2% ─────────────────────────
    def neg_aa(w):  return -(act_alpha @ w)
    def neg_aa_g(_): return -act_alpha

    def te_slack(w):
        te, g = te_and_grad(w, B, b_spy, sigma_F, d)
        return TARGET_TE - te, -g

    res1 = run_opt(
        neg_aa, neg_aa_g,
        constraints=[eq_con,
                     {"type": "ineq",
                      "fun": lambda w: te_slack(w)[0],
                      "jac": lambda w: te_slack(w)[1]}],
        bounds=bounds, w0=w_warm.copy(),
        label="Maximise alpha | TE <= 2%",
    )
    w1 = np.clip(res1.x, 0, MAX_WEIGHT); w1 /= w1.sum()
    r1 = report("Strategy 1: Max-Alpha (TE <= 2%)",
                w1, tickers, B, b_spy, sigma_F, d, act_alpha, betas)

    # ── Strategy 2: minimise TE | active alpha >= 1% ─────────────────────────
    def te_obj(w): return te_and_grad(w, B, b_spy, sigma_F, d)

    def aa_slack(w):   return (act_alpha @ w) - TARGET_ALPHA
    def aa_slack_g(_): return act_alpha

    res2 = run_opt(
        te_obj, True,
        constraints=[eq_con,
                     {"type": "ineq",
                      "fun": aa_slack, "jac": aa_slack_g}],
        bounds=bounds, w0=w_warm.copy(),
        label="Minimise TE | active alpha >= 1%",
    )
    w2 = np.clip(res2.x, 0, MAX_WEIGHT); w2 /= w2.sum()
    r2 = report("Strategy 2: Min-TE (active alpha >= 1%)",
                w2, tickers, B, b_spy, sigma_F, d, act_alpha, betas)

    # ── Save ─────────────────────────────────────────────────────────────────
    data_dir = Path(__file__).parent.parent / "data"
    pd.DataFrame([r1, r2]).to_csv(
        data_dir / "optimised_portfolios_summary.csv", index=False)

    for label, w in [("max_alpha", w1), ("min_te", w2)]:
        df_w = pd.DataFrame({
            "ticker":       tickers,
            "weight_%":     (w * 100).round(4),
            "act_alpha_%":  (act_alpha * 100).round(2),
            "mkt_beta":     betas.loc[tickers, "Mkt-RF"].values.round(3),
        })
        df_w = df_w[df_w["weight_%"] > 0.01].sort_values("weight_%", ascending=False)
        df_w.to_csv(data_dir / f"portfolio_weights_{label}.csv", index=False)

    print(f"\nSaved -> data/optimised_portfolios_summary.csv")
    print(f"Saved -> data/portfolio_weights_max_alpha.csv  "
          f"({(w1 > 0.001).sum()} stocks)")
    print(f"Saved -> data/portfolio_weights_min_te.csv  "
          f"({(w2 > 0.001).sum()} stocks)")


if __name__ == "__main__":
    main()
