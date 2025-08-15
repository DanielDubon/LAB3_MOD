# optimizers_safe.py
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional

Array = np.ndarray
ScalarFunc = Callable[[Array], float]
GradFunc   = Callable[[Array], Array]
HessFunc   = Callable[[Array], Array]

# ---------------- utilidades ----------------

class StopCriterion:
    GRAD_NORM   = "grad_norm"
    STEP_NORM   = "step_norm"
    REL_F_CHANGE= "rel_f_change"

def _should_stop(crit: str, eps: float,
                 x_prev: Array, x_cur: Array,
                 f_prev: float, f_cur: float,
                 g_cur: Array) -> Tuple[bool, float]:
    if crit == StopCriterion.GRAD_NORM:
        err = np.linalg.norm(g_cur)
        return err <= eps, err
    if crit == StopCriterion.STEP_NORM:
        err = np.linalg.norm(x_cur - x_prev)
        return err <= eps, err
    if crit == StopCriterion.REL_F_CHANGE:
        denom = max(1.0, abs(f_prev))
        err = abs(f_cur - f_prev) / denom
        return err <= eps, err
    raise ValueError(f"Criterio desconocido: {crit}")

def _result(method: str, x_seq, f_seq, err_seq, conv: bool) -> Dict:
    return {
        "method": method, "best": x_seq[-1].copy(),
        "x_seq": x_seq, "f_seq": f_seq, "err_seq": err_seq,
        "n_iter": len(f_seq) - 1, "converged": conv
    }

def _armijo(
    f: ScalarFunc, df: GradFunc, x: Array, d: Array,
    f_x: float, g_x: Array, alpha0: float,
    c: float = 1e-4, shrink: float = 0.5,
    min_alpha: float = 1e-12, max_trials: int = 50
) -> Tuple[Array, float, float, bool]:
    """Siempre intenta asegurar f(x+αd) <= f(x) + c α <g,d> y finitud."""
    gdotd = float(g_x @ d)
    a = alpha0
    for _ in range(max_trials):
        x_new = x + a * d
        if not np.all(np.isfinite(x_new)):
            a *= shrink; 
            if a < min_alpha: break
            continue
        f_new = f(x_new)
        if np.isfinite(f_new) and f_new <= f_x + c * a * gdotd:
            return x_new, f_new, a, True
        a *= shrink
        if a < min_alpha: break
    return x, f_x, 0.0, False

# ---------------- 1) GD dirección aleatoria ----------------

def gd_random_direction(
    f: ScalarFunc, df: GradFunc, x0: Array,
    alpha: float = 0.05, max_iter: int = 500,
    eps: float = 1e-6, stop: str = StopCriterion.GRAD_NORM,
    rng: Optional[np.random.Generator] = None
) -> Dict:
    if rng is None: rng = np.random.default_rng()
    x = x0.astype(float)
    x_seq, f_seq, err_seq = [x.copy()], [f(x)], [np.inf]
    conv = False
    for _ in range(max_iter):
        g = df(x)
        u = rng.normal(size=x.shape); u /= (np.linalg.norm(u)+1e-15)
        d = -u if float(g @ u) > 0 else u
        x_new, f_new, _, ok = _armijo(f, df, x, d, f_seq[-1], g, alpha)
        if not ok: break
        stop_now, err = _should_stop(stop, eps, x, x_new, f_seq[-1], f_new, df(x_new))
        x = x_new
        x_seq.append(x.copy()); f_seq.append(f_new); err_seq.append(err)
        if stop_now: conv = True; break
    return _result("gd_random_direction", x_seq, f_seq, err_seq, conv)

# ---------------- 2) Steepest descent ----------------

def steepest_descent(
    f: ScalarFunc, df: GradFunc, x0: Array,
    alpha: float = 0.1, max_iter: int = 500,
    eps: float = 1e-6, stop: str = StopCriterion.GRAD_NORM
) -> Dict:
    x = x0.astype(float)
    x_seq, f_seq, err_seq = [x.copy()], [f(x)], [np.inf]
    conv = False
    for _ in range(max_iter):
        g = df(x); d = -g
        x_new, f_new, _, ok = _armijo(f, df, x, d, f_seq[-1], g, alpha)
        if not ok: break
        stop_now, err = _should_stop(stop, eps, x, x_new, f_seq[-1], f_new, df(x_new))
        x = x_new
        x_seq.append(x.copy()); f_seq.append(f_new); err_seq.append(err)
        if stop_now: conv = True; break
    return _result("steepest_descent", x_seq, f_seq, err_seq, conv)

# ---------------- 3) Newton (Hessiano exacto) ----------------

def newton(
    f: ScalarFunc, df: GradFunc, ddf: HessFunc, x0: Array,
    alpha: float = 1.0, max_iter: int = 100, eps: float = 1e-6,
    stop: str = StopCriterion.GRAD_NORM, damping: bool = False
) -> Dict:
    x = x0.astype(float)
    x_seq, f_seq, err_seq = [x.copy()], [f(x)], [np.inf]
    conv = False
    for _ in range(max_iter):
        g = df(x); H = ddf(x)
        try: p = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            p = -np.linalg.solve(H + 1e-6*np.eye(H.shape[0]), g)
        step_scale = (alpha if damping else 1.0)
        x_new, f_new, _, ok = _armijo(f, df, x, p, f_seq[-1], g, step_scale)
        if not ok: break
        stop_now, err = _should_stop(stop, eps, x, x_new, f_seq[-1], f_new, df(x_new))
        x = x_new
        x_seq.append(x.copy()); f_seq.append(f_new); err_seq.append(err)
        if stop_now: conv = True; break
    return _result("newton_exact", x_seq, f_seq, err_seq, conv)

# ---------------- 4) Gradiente conjugado no lineal ----------------

def conjugate_gradient(
    f: ScalarFunc, df: GradFunc, x0: Array,
    alpha: float = 0.1, variant: str = "PR",
    max_iter: int = 1000, eps: float = 1e-6,
    stop: str = StopCriterion.GRAD_NORM,
    restart_each: Optional[int] = None,
    beta_clip: Tuple[float, float] = (-1e3, 1e3)
) -> Dict:
    def beta(g_new: Array, g_old: Array, d_old: Array) -> float:
        if variant == "FR":
            num = float(g_new @ g_new); den = float(g_old @ g_old) + 1e-15
            b = num/den
        elif variant == "PR":
            y = g_new - g_old
            num = float(g_new @ y); den = float(g_old @ g_old) + 1e-15
            b = max(0.0, num/den)   # PR+
        elif variant == "HS":
            y = g_new - g_old
            num = float(g_new @ y); den = float(d_old @ y) + 1e-15
            b = num/den
        else:
            raise ValueError("variant in {'FR','PR','HS'}")
        return float(np.clip(b, beta_clip[0], beta_clip[1]))

    x = x0.astype(float)
    g = df(x); d = -g
    x_seq, f_seq, err_seq = [x.copy()], [f(x)], [np.inf]
    conv = False
    for k in range(max_iter):
        # SIEMPRE Armijo con paso inicial alpha
        x_try, f_try, a_used, ok = _armijo(f, df, x, d, f_seq[-1], g, alpha)
        if not ok: break
        g_new = df(x_try)

        stop_now, err = _should_stop(stop, eps, x, x_try, f_seq[-1], f_try, g_new)
        x_prev, g_prev = x, g
        x, g = x_try, g_new
        x_seq.append(x.copy()); f_seq.append(f_try); err_seq.append(err)
        if stop_now: conv = True; break

        if restart_each is not None and (k+1) % restart_each == 0:
            d = -g
        else:
            b = beta(g, g_prev, d)
            d = -g + b*d
            if float(d @ g) >= 0:   # mantener dirección de descenso
                d = -g
    return _result(f"nonlinear_CG_{variant}", x_seq, f_seq, err_seq, conv)

# ---------------- 5) BFGS ----------------

def bfgs(
    f: ScalarFunc, df: GradFunc, x0: Array,
    alpha: float = 0.2, max_iter: int = 1000,
    eps: float = 1e-6, stop: str = StopCriterion.GRAD_NORM
) -> Dict:
    n = x0.size
    x = x0.astype(float)
    H = np.eye(n)
    g = df(x)
    x_seq, f_seq, err_seq = [x.copy()], [f(x)], [np.inf]
    conv = False
    for _ in range(max_iter):
        p = -H @ g
        x_new, f_new, _, ok = _armijo(f, df, x, p, f_seq[-1], g, alpha)
        if not ok: break
        g_new = df(x_new)

        s = (x_new - x).reshape(-1,1)
        y = (g_new - g).reshape(-1,1)
        sty = float((s.T @ y).item())
        if sty > 1e-12:
            rho = 1.0 / sty
            I = np.eye(n)
            V = (I - rho * (s @ y.T))
            H = V @ H @ V.T + rho * (s @ s.T)
            # asegurar SPD si hace falta
            try: np.linalg.cholesky(H)
            except np.linalg.LinAlgError: H += 1e-10*np.eye(n)

        stop_now, err = _should_stop(stop, eps, x, x_new, f_seq[-1], f_new, g_new)
        x, g = x_new, g_new
        x_seq.append(x.copy()); f_seq.append(f_new); err_seq.append(err)
        if stop_now: conv = True; break
    return _result("BFGS", x_seq, f_seq, err_seq, conv)

# ---------------- demo mínimo ----------------
if __name__ == "__main__":
    A = np.diag([1.0, 10.0])
    def f_quadratic(x: Array) -> float:
        z = x - np.array([1.0, -2.0])
        return float((z.T @ A @ z).item())
    def df_quadratic(x: Array) -> Array:
        z = x - np.array([1.0, -2.0])
        return 2 * (A @ z)
    def ddf_quadratic(x: Array) -> Array: return 2 * A
    x0 = np.array([10.0, 10.0])

    print("SD:",      steepest_descent(f_quadratic, df_quadratic, x0, alpha=0.5)["best"])
    print("Newton:",  newton(f_quadratic, df_quadratic, ddf_quadratic, x0)["best"])
    print("BFGS:",    bfgs(f_quadratic, df_quadratic, x0, alpha=1.0)["best"])
    print("CG(PR):",  conjugate_gradient(f_quadratic, df_quadratic, x0, alpha=1.0, variant="PR")["best"])
    print("GD rand:", gd_random_direction(f_quadratic, df_quadratic, x0, alpha=0.5)["best"])
