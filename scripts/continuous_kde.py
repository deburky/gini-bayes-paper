"""Continuous KDE-CAP example for the Gini-Bayes paper.

Generates a synthetic continuous-score portfolio, fits class-conditional
kernel density estimates f(x|D), f(x|~D), f(x), and shows that the density
route recovers (i) the weight of evidence WOE(x) = ln f(x|D) - ln f(x|~D),
(ii) the standardized PD p(D|x) = p_D f(x|D)/f(x), and (iii) the CAP / Gini.

Writes pgfplots-ready .dat files into data/ and prints the scalar results.
Run: uv run python scripts/continuous_kde.py
"""

import os

import numpy as np
from scipy.stats import gaussian_kde


def sig(z):
    """Logistic sigmoid 1 / (1 + e^-z)."""
    return 1 / (1 + np.exp(-z))


def logit(p):
    """Log-odds ln(p / (1 - p))."""
    return np.log(p / (1 - p))


rng = np.random.default_rng(7)

# ------------------------------------------------------------------------------
# Data-Generating Process: score x, logistic truth
# ------------------------------------------------------------------------------
n = 8000
b0, b1 = -1.3, 1.1  # true log-odds = b0 + b1 * x
x = rng.normal(0, 1, n)  # standardized score (higher = worse)
p_true = sig(b0 + b1 * x)
y = rng.binomial(1, p_true)
pD = y.mean()
xb, xg = x[y == 1], x[y == 0]  # bad / good scores

# ------------------------------------------------------------------------------
# Class-Conditional KDEs
# ------------------------------------------------------------------------------
fD = gaussian_kde(xb)  # f(x|D)
fG = gaussian_kde(xg)  # f(x|~D)
fX = gaussian_kde(x)  # f(x)

# evaluation grid over the bulk of the data
lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
grid = np.linspace(lo, hi, 60)
dD, dG, dX = fD(grid), fG(grid), fX(grid)

woe_kde = np.log(dD) - np.log(dG)  # WOE(x) from densities
woe_true = (b0 + b1 * grid) - logit(pD)  # true WOE = logit(p)-logit(pD)
phat = pD * dD / dX  # standardized PD route
p_grid_true = sig(b0 + b1 * grid)

# recover the logit slope by OLS on the KDE WOE over the dense central region
mask = (grid > -2) & (grid < 2)
slope, intercept = np.polyfit(grid[mask], woe_kde[mask], 1)

# ------------------------------------------------------------------------------
# Continuous CAP + Gini (sort worst-first by score)
# ------------------------------------------------------------------------------
o = np.argsort(-x)
Fx = np.arange(1, n + 1) / n
FxD = np.cumsum(y[o]) / y.sum()
A = np.trapezoid(np.r_[0, FxD], np.r_[0, Fx])
gini_cap = (2 * A - 1) / (1 - pD)
# cross-check via Mann-Whitney AUC = P(score_bad > score_good)
rank = np.argsort(np.argsort(x)) + 1  # ranks 1..n
auc = (rank[y == 1].sum() - xb.size * (xb.size + 1) / 2) / (xb.size * xg.size)
gini_auc = 2 * auc - 1

# ------------------------------------------------------------------------------
# Write pgfplots .dat Files
# ------------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)


def dump(name, cols, header):
    """Write columns to data/<name>.dat as a pgfplots-ready table."""
    np.savetxt(
        f"data/{name}.dat",
        np.column_stack(cols),
        header=header,
        comments="",
        fmt="%.5f",
    )


dump("kde_densities", [grid, dD, dG, dX], "x fD fG fX")
dump("kde_woe", [grid, woe_kde, woe_true], "x woe_kde woe_true")
dump("kde_pd", [grid, phat, p_grid_true], "x phat ptrue")
# subsample CAP to ~80 points for a light .dat
idx = np.linspace(0, n - 1, 80).astype(int)
dump("kde_cap", [np.r_[0, Fx[idx]], np.r_[0, FxD[idx]]], "Fx FxD")

print(f"n={n}  p_D={pD:.4f}")
print(f"CAP area A={A:.4f}  Gini(CAP)={gini_cap:.4f}  Gini(2AUC-1)={gini_auc:.4f}")
print(
    f"true logit slope b1={b1}  recovered from KDE WOE={slope:.4f}  (intercept {intercept:.3f})"
)
print(f"true intercept of WOE line = b0 - logit(pD) = {b0 - logit(pD):.4f}")
print("wrote data/kde_densities.dat kde_woe.dat kde_pd.dat kde_cap.dat")
