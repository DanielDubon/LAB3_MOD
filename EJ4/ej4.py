import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("datos_lab3.csv")
x = df["x"].to_numpy()
y = df["y"].to_numpy()
n = len(y)

def design_matrix(x):
    return np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        np.sin(7*x),
        np.sin(13*x),
    ])

X = design_matrix(x)

D = np.zeros((n-1, n))
for i in range(n-1):
    D[i, i]   = -1.0
    D[i, i+1] =  1.0


def fit_beta(X, y, lam):
    XtX = X.T @ X
    LtL = X.T @ (D.T @ D) @ X
    A = XtX + lam * LtL
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    return beta

lambdas = [0.0, 100.0, 500.0]
betas = {lam: fit_beta(X, y, lam) for lam in lambdas}

xg = np.linspace(x.min(), x.max(), 800)
Xg = design_matrix(xg)
yg = {lam: Xg @ betas[lam] for lam in lambdas}

plt.figure(figsize=(10,6))
plt.scatter(x, y, s=12, label="Datos", alpha=0.8)
for lam, style in zip(lambdas, ["-", "--", "-."]):
    plt.plot(xg, yg[lam], linestyle=style, linewidth=2, label=fr"$\lambda={lam:g}$")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Regresion con penalizacion de suavidad (Tikhonov en diferencias)")
plt.grid(True); plt.legend(); plt.tight_layout()

out = "ej4_result.png"
plt.savefig(out, dpi=200)
print(f"Figura guardada en: {out}")

print("B por l:")
for lam in lambdas:
    print(f"  l={lam:>5g} -> {betas[lam]}")
