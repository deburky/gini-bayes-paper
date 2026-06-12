"""Generate a standalone pgfplots figure (paper format) comparing two models:
same ranking / same Gini, different calibration. Uses the paper's notation
(observed E[Y|x], predicted p_hat(D|x)) and the Gini-gap as the calibration
summary. Writes figures/compare-two-models.tex.

Run: uv run python scripts/make_compare_figure.py
"""
import numpy as np

sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)

n = 15000
z = rng.normal(-1.4, 1.3, n)
y = rng.binomial(1, sig(z))
pD = y.mean()

pA = sig(z)                                  # Model A: claims the truth
beta = 1.8                                   # Model B: overconfident
b0g = np.linspace(-3, 3, 601)
b0 = b0g[np.argmin([abs(sig(b + beta * z).mean() - pD) for b in b0g])]
pB = sig(b0 + beta * z)

o = np.argsort(-z)
Fx = np.arange(1, n + 1) / n
Femp = np.cumsum(y[o]) / y.sum()
FmodA = np.cumsum(pA[o]) / pA.sum()
FmodB = np.cumsum(pB[o]) / pB.sum()

def gini(FD, p):
    return (2 * np.trapezoid(np.r_[0, FD], np.r_[0, Fx]) - 1) / (1 - p)

g_emp, g_A, g_B = gini(Femp, pD), gini(FmodA, pA.mean()), gini(FmodB, pB.mean())

def reliab(p):
    e = np.quantile(p, np.linspace(0, 1, 11)); e[-1] += 1e-9
    idx = np.digitize(p, e[1:-1]); xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum(): xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)

xA, yA = reliab(pA); xB, yB = reliab(pB)

def coords(xs, ys, k=1):
    s = np.linspace(0, len(xs) - 1, min(len(xs), 60)).astype(int) if k > 1 else range(len(xs))
    return " ".join(f"({xs[i]:.4f},{ys[i]:.4f})" for i in s)

idx = np.linspace(0, n - 1, 55).astype(int)
cap_emp = "(0,0) " + coords(Fx[idx], Femp[idx])
cap_A = "(0,0) " + coords(Fx[idx], FmodA[idx])
cap_B = "(0,0) " + coords(Fx[idx], FmodB[idx])
rel_A = coords(xA, yA); rel_B = coords(xB, yB)

tex = r"""\documentclass[border=6pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{amsmath}
\usepackage{amssymb}
\newcommand{\E}{\mathbb{E}}
\definecolor{capblue}{HTML}{2C6FB5}
\definecolor{capred}{HTML}{C0392B}
\definecolor{capgray}{HTML}{7F8C8D}
\definecolor{capdark}{HTML}{2C3E50}
\begin{document}
\begin{tikzpicture}

% ===== Panel 1: CAP (ranking) =====
\begin{axis}[
  name=cap, width=8.2cm, height=7.6cm, xmin=0,xmax=1,ymin=0,ymax=1.02,
  xlabel={$F(x)$ --- borrowers (worst first)}, ylabel={$F(x\mid D)$ --- defaults},
  title={\textbf{Area: same Gini for both models}},
  title style={font=\small,yshift=2pt}, tick label style={font=\footnotesize},
  label style={font=\small}, grid=major, grid style={gray!15},
  legend style={at={(0.97,0.03)},anchor=south east,draw=none,font=\scriptsize},
]
\addplot[capgray,dotted,thick] coordinates {(0,0)(1,1)}; \addlegendentry{random}
\addplot[capdark,very thick] coordinates {__CAP_EMP__}; \addlegendentry{empirical (outcomes)}
\addplot[capblue,very thick,dashed] coordinates {__CAP_A__}; \addlegendentry{Model A claims}
\addplot[capred,very thick,dashed] coordinates {__CAP_B__}; \addlegendentry{Model B claims}
\node[font=\scriptsize,anchor=north west,align=left] at (axis cs:0.04,0.99)
 {empirical Gini $=__GEMP__$\\[1pt](same ranking $\Rightarrow$ same Gini)\\[3pt]
  \textcolor{capblue}{A claims $__GA__$ (calibrated)}\\
  \textcolor{capred}{B claims $__GB__$ (inflated)}};
\end{axis}

% ===== Panel 2: reliability (calibration) =====
\begin{axis}[
  name=rel, at={(cap.east)}, anchor=west, xshift=1.5cm,
  width=8.2cm, height=7.6cm, xmin=0,xmax=0.85,ymin=0,ymax=0.85,
  xlabel={predicted $\hat p(D\mid x)$}, ylabel={observed $\E[Y\mid x]$},
  title={\textbf{Slopes: calibration tells them apart}},
  title style={font=\small,yshift=2pt}, tick label style={font=\footnotesize},
  label style={font=\small}, grid=major, grid style={gray!15},
  legend style={at={(0.03,0.97)},anchor=north west,draw=none,font=\scriptsize},
]
\addplot[capgray,dotted,thick] coordinates {(0,0)(0.85,0.85)}; \addlegendentry{calibrated ($45^\circ$)}
\addplot[capblue,thick,mark=*,mark size=1.6pt] coordinates {__REL_A__}; \addlegendentry{Model A}
\addplot[capred,thick,mark=square*,mark size=1.6pt] coordinates {__REL_B__}; \addlegendentry{Model B (overconfident)}
\node[capred,font=\scriptsize,anchor=south east,align=right] at (axis cs:0.83,0.04)
 {claims too extreme:\\over at the top,\\under at the bottom};
\end{axis}

\end{tikzpicture}
\end{document}
"""
repl = {
    "__CAP_EMP__": cap_emp, "__CAP_A__": cap_A, "__CAP_B__": cap_B,
    "__REL_A__": rel_A, "__REL_B__": rel_B,
    "__GEMP__": f"{g_emp:.2f}", "__GA__": f"{g_A:.2f}", "__GB__": f"{g_B:.2f}",
}
for k, v in repl.items():
    tex = tex.replace(k, v)
open("figures/compare-two-models.tex", "w").write(tex)

# ---- data files for embedding in the paper (\addplot table) ----
import os
os.makedirs("data", exist_ok=True)
cap = np.column_stack([np.r_[0, Fx[idx]], np.r_[0, Femp[idx]],
                       np.r_[0, FmodA[idx]], np.r_[0, FmodB[idx]]])
np.savetxt("data/compare_cap.dat", cap, header="Fx Femp FmodA FmodB",
           comments="", fmt="%.5f")
rel = np.column_stack([xA, yA, xB, yB])
np.savetxt("data/compare_rel.dat", rel, header="pA EA pB EB",
           comments="", fmt="%.5f")

print(f"pD={pD:.3f} Gini_emp={g_emp:.3f} Gini_A={g_A:.3f} Gini_B={g_B:.3f}")
print("wrote figures/compare-two-models.tex, data/compare_cap.dat, data/compare_rel.dat")
