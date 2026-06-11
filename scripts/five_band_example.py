"""Discrete five-band example for the Gini-Bayes paper.

Reproduces every number in the running example: the Bayes reweighting, the
accuracy ratio computed three ways (CAP area, Somers' D, 2*AUC-1), the weight
of evidence / information value, and the comparison-mode calibration gap.

Run: uv run python scripts/five_band_example.py
"""

import numpy as np

# --- five-band portfolio (Table 1) ---
count = np.array([24, 36, 25, 20, 10.0])
bad = np.array([1, 4, 5, 5, 5.0])
good = count - bad
N, B, G = count.sum(), bad.sum(), good.sum()
pDx, fx, pD = bad / count, count / N, B / N  # p(D|x), f(x), prior p_D
fxD, fxG = bad / B, good / G  # f(x|D), f(x|~D)  -- Bayes
order = [4, 3, 2, 1, 0]  # worst-first: E, D, C, B, A
Fx, FxD = np.cumsum(fx[order]), np.cumsum(fxD[order])

# --- accuracy ratio three ways (all = 0.4395) ---
A = np.trapezoid(np.r_[0, FxD], np.r_[0, Fx])  # CAP area = 0.6815
AR = (2 * A - 1) / (1 - pD)  # exact (naive 2A-1 = 0.363)

risk = dict(zip("ABCDE", range(5)))
names = "ABCDE"
P = Q = T = 0  # Kendall: P concordant, Q discordant, T tied
for i in range(5):
    for j in range(5):
        if risk[names[j]] > risk[names[i]]:
            P += good[i] * bad[j]
        elif risk[names[j]] < risk[names[i]]:
            Q += good[i] * bad[j]
        else:
            T += good[i] * bad[j]
somers = (P - Q) / (G * B)  # Somers' D_yx = 0.4395 = AR
auc = (P + 0.5 * T) / (G * B)  # 0.7197; 2*auc - 1 = AR

# --- weight of evidence and information value ---
woe = np.log(fxD / fxG)  # bad-over-good convention
IV = np.sum((fxD - fxG) * woe)  # 0.731

# --- comparison mode: compressed-log-odds model (Sec. 6) ---
sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(p / (1 - p))
phat = sig(-0.4 + 0.6 * logit(pDx))
phatD = (fx * phat).sum()
FmodD = np.cumsum(((phat * fx) / phatD)[order])
Amod = np.trapezoid(np.r_[0, FmodD], np.r_[0, Fx])
gini_emp = (2 * A - 1) / (1 - pD)  # 0.4395
gini_mod = (2 * Amod - 1) / (1 - phatD)  # 0.2823

if __name__ == "__main__":
    print(f"p_D = {pD:.4f}  CAP area A = {A:.4f}")
    print(f"AR (CAP) = {AR:.4f}  Somers' D = {somers:.4f}  2*AUC-1 = {2 * auc - 1:.4f}")
    print(f"IV = {IV:.4f}")
    print(f"Gini empirical = {gini_emp:.4f}  Gini model = {gini_mod:.4f}")
