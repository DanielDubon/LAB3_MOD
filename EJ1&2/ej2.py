# run_lab3.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ej1 import (
    steepest_descent, gd_random_direction, newton, conjugate_gradient, bfgs
)

# ---------- Funciones del enunciado ----------
# 2a
def f_2a(xy):
    x, y = xy
    return float(x**4 + y**4 - 4*x*y + 0.5*y + 1.0)
def df_2a(xy):
    x, y = xy
    return np.array([4*x**3 - 4*y, 4*y**3 - 4*x + 0.5], float)
def ddf_2a(xy):
    x, y = xy
    return np.array([[12*x**2, -4], [-4, 12*y**2]], float)

# 2b Rosenbrock 2D
def f_rosen2(xy):
    x1, x2 = xy
    return float(100*(x2 - x1**2)**2 + (1 - x1)**2)
def df_rosen2(xy):
    x1, x2 = xy
    return np.array([-400*x1*(x2 - x1**2) - 2*(1 - x1),
                      200*(x2 - x1**2)], float)
def ddf_rosen2(xy):
    x1, x2 = xy
    return np.array([[1200*x1**2 - 400*x2 + 2, -400*x1],
                     [-400*x1, 200]], float)

# 2c Rosenbrock 7D
def f_rosen7(x):
    s = 0.0
    for i in range(6):
        s += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return float(s)
def df_rosen7(x):
    n = 7
    g = np.zeros(n, float)
    for i in range(n-1):
        g[i]   += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
        g[i+1] +=  200*(x[i+1] - x[i]**2)
    return g

# ---------- helpers ----------
def run_and_row(name, method_fn, f, df, ddf, x0, xstar=None):
    res = method_fn(f, df, ddf, x0)
    x_best = res["best"]; g_norm = np.linalg.norm(df(x_best))
    err = np.linalg.norm(x_best - xstar) if xstar is not None else np.nan
    return {
        "Problema": name,
        "Algoritmo": res["method"],
        "Convergencia": "Sí" if res["converged"] else "No",
        "Iteraciones": res["n_iter"],
        "f(x*)": f(x_best),
        "||grad||": g_norm,
        "x* aproximado": np.array2string(x_best, precision=6),
        "Error vs óptimo": err
    }, res["x_seq"]

def plot_contours_with_path(f, path, xlim, ylim, title, outfile):
    xs = np.linspace(*xlim, 200); ys = np.linspace(*ylim, 200)
    X, Y = np.meshgrid(xs, ys); Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = f(np.array([X[i,j], Y[i,j]]))
    plt.figure()
    plt.contour(X, Y, Z, 30)
    p = np.vstack(path); plt.plot(p[:,0], p[:,1], marker='o', linewidth=1)
    plt.title(title); plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
    plt.savefig(outfile, dpi=160); plt.close()

def plot_error(errs, title, outfile):
    ks = np.arange(len(errs))
    plt.figure()
    plt.plot(ks, errs, marker='o', linewidth=1)
    plt.xlabel("Iteración k"); plt.ylabel("Error ||x_k - x*||")
    plt.title(title); plt.tight_layout()
    plt.savefig(outfile, dpi=160); plt.close()

# ---------- Experimentos ----------
if __name__ == "__main__":
    results = []

    # Métodos (α elegidos para converger bien)
    methods_2d = [
        ("SD",      lambda f,df,ddf,x0: steepest_descent(f,df,x0,alpha=0.01, max_iter=20000)),
        ("RandGD",  lambda f,df,ddf,x0: gd_random_direction(f,df,x0,alpha=0.1, max_iter=5000)),
        ("Newton",  lambda f,df,ddf,x0: newton(f,df,ddf,x0,alpha=1.0, max_iter=200)),
        ("CG-PR",   lambda f,df,ddf,x0: conjugate_gradient(f,df,x0,alpha=0.5, variant="PR", max_iter=10000, restart_each=200)),
        ("BFGS",    lambda f,df,ddf,x0: bfgs(f,df,x0,alpha=1.0, max_iter=20000)),
    ]

    # ---- 2a
    x0 = np.array([-3.0, 1.0]); xstar = np.array([-1.01463, -1.04453])
    for name, runner in methods_2d:
        row, xseq = run_and_row("2a", runner, f_2a, df_2a, ddf_2a, x0, xstar)
        results.append(row)
        # gráficos
        plot_contours_with_path(f_2a, xseq, (-3.5,1.5), (-3.0,2.0), f"2a: {row['Algoritmo']}", f"2a_{name}_contour.png")
        errs = [np.linalg.norm(x - xstar) for x in xseq]
        plot_error(errs, f"2a: Error – {row['Algoritmo']}", f"2a_{name}_error.png")

    # ---- 2b
    x0 = np.array([-1.2, 1.0]); xstar = np.array([1.0, 1.0])
    for name, runner in methods_2d:
        row, xseq = run_and_row("2b", runner, f_rosen2, df_rosen2, ddf_rosen2, x0, xstar)
        results.append(row)
        plot_contours_with_path(f_rosen2, xseq, (-1.5,2.0), (-0.5,2.0), f"2b: {row['Algoritmo']}", f"2b_{name}_contour.png")
        errs = [np.linalg.norm(x - xstar) for x in xseq]
        plot_error(errs, f"2b: Error – {row['Algoritmo']}", f"2b_{name}_error.png")

    # ---- 2c (7D): SD, CG-PR, BFGS
    methods_hi = [
        ("SD",    lambda f,df,x0: steepest_descent(f,df,x0,alpha=0.001, max_iter=200000)),
        ("CG-PR", lambda f,df,x0: conjugate_gradient(f,df,x0,alpha=0.1, variant="PR", max_iter=200000, restart_each=500)),
        ("BFGS",  lambda f,df,x0: bfgs(f,df,x0,alpha=0.5, max_iter=200000)),
    ]
    x0 = np.array([-1.2, 1.0, 1.0, 1.0, 1.0, -1.2, 1.0])
    for name, runner in methods_hi:
        res = runner(f_rosen7, df_rosen7, x0)
        x_best = res["best"]; g_norm = np.linalg.norm(df_rosen7(x_best))
        err = np.linalg.norm(x_best - np.ones(7))
        results.append({
            "Problema": "2c",
            "Algoritmo": res["method"],
            "Convergencia": "Sí" if res["converged"] else "No",
            "Iteraciones": res["n_iter"],
            "f(x*)": f_rosen7(x_best),
            "||grad||": g_norm,
            "x* aproximado": np.array2string(x_best, precision=6),
            "Error vs óptimo": err
        })

    # ---- Exportar tabla CSV
    df = pd.DataFrame(results)
    df.to_csv("resultados_optimizacion_lab3.csv", index=False)
    print("OK -> resultados_optimizacion_lab3.csv y PNGs generados.")
