"""Statistical machinery: bootstrap CIs, logistic thresholds, survival,
run-level cluster bootstrap.

Everything reports dispersion across runs; no single-trajectory claims.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def bootstrap_ci(values, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI for a statistic over independent runs."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0, "estimate": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(len(v), size=len(v))])
                      for _ in range(n_boot)])
    return {"n": int(len(v)), "estimate": float(stat(v)),
            "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2))}


def binomial_ci(k: int, n: int, alpha: float = 0.05) -> dict:
    """Wilson interval for a proportion."""
    if n == 0:
        return {"n": 0, "p": None, "lo": None, "hi": None}
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return {"n": int(n), "k": int(k), "p": float(p),
            "lo": float(max(0, centre - half)),
            "hi": float(min(1, centre + half))}


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """MLE fit of P(y=1) = sigmoid(a + b*x); returns (a, b) or None."""
    def nll(params):
        a, b = params
        z = a + b * x
        return float(np.sum(np.logaddexp(0, z)) - np.sum(y * z))
    for b0 in (1.0, -1.0, 5.0, -5.0):
        res = optimize.minimize(nll, x0=[0.0, b0], method="Nelder-Mead",
                                options={"maxiter": 2000, "xatol": 1e-6})
        if res.success:
            return float(res.x[0]), float(res.x[1])
    return None


def logistic_threshold(x_values, successes, n_boot: int = 2000,
                       seed: int = 0, log_x: bool = False) -> dict:
    """Fit flip/survival probability vs x; return x at p=0.5 with a
    bootstrap CI.  ``x_values``/``successes`` are per-run (one row per run,
    success in {0,1}); bootstrap resamples runs within each x level."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(successes, dtype=float)
    xt = np.log(x) if log_x else x

    def x50_of(xs, ys):
        fit = _fit_logistic(xs, ys)
        if fit is None:
            return None
        a, b = fit
        if abs(b) < 1e-9:
            return None
        val = -a / b
        return float(np.exp(val)) if log_x else float(val)

    est = x50_of(xt, y)
    rng = np.random.default_rng(seed)
    levels = np.unique(x)
    idx_by_level = {lv: np.flatnonzero(x == lv) for lv in levels}
    boots = []
    for _ in range(n_boot):
        take = np.concatenate([
            idx[rng.integers(len(idx), size=len(idx))]
            for idx in idx_by_level.values()])
        v = x50_of(xt[take], y[take])
        if v is not None and np.isfinite(v):
            boots.append(v)
    boots = np.asarray(boots)
    # clip pathological resamples (threshold outside observed range x10)
    if len(boots):
        lo_lim, hi_lim = x.min() / 10, x.max() * 10
        boots = boots[(boots > lo_lim) & (boots < hi_lim)]
    return {
        "x50": est,
        "lo": float(np.quantile(boots, 0.025)) if len(boots) else None,
        "hi": float(np.quantile(boots, 0.975)) if len(boots) else None,
        "n_runs": int(len(y)),
        "n_boot_kept": int(len(boots)),
    }


def km_survival(durations, events) -> dict:
    """Kaplan-Meier estimator.  durations: time of event or censoring;
    events: 1 = convention lost, 0 = censored (still alive at end)."""
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)
    order = np.argsort(d)
    d, e = d[order], e[order]
    times, surv = [0.0], [1.0]
    at_risk = len(d)
    s = 1.0
    for t in np.unique(d):
        deaths = int(np.sum((d == t) & (e == 1)))
        n_at = int(np.sum(d >= t))
        if deaths > 0 and n_at > 0:
            s *= 1 - deaths / n_at
            times.append(float(t))
            surv.append(float(s))
    return {"times": times, "survival": surv, "n": int(len(d)),
            "n_events": int(e.sum())}


def cluster_bootstrap_slope(run_ids, x, y, n_boot: int = 5000,
                            seed: int = 0) -> dict:
    """OLS slope of y on x with resampling at the RUN level (repeated
    measures within a run are not independent).  Used for the
    normative-share vs convention-age trend."""
    run_ids = np.asarray(run_ids)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    runs = np.unique(run_ids)
    idx_by_run = {r: np.flatnonzero(run_ids == r) for r in runs}

    def slope(idxs):
        xv, yv = x[idxs], y[idxs]
        if len(xv) < 3 or np.var(xv) == 0:
            return None
        return float(np.polyfit(xv, yv, 1)[0])

    est = slope(np.arange(len(x)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = runs[rng.integers(len(runs), size=len(runs))]
        idxs = np.concatenate([idx_by_run[r] for r in chosen])
        s = slope(idxs)
        if s is not None:
            boots.append(s)
    boots = np.asarray(boots)
    if len(boots) == 0:
        return {"slope": est, "lo": None, "hi": None, "p_two_sided": None,
                "n_runs": int(len(runs))}
    p = 2 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
    return {"slope": est,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
            "p_two_sided": min(1.0, p),
            "n_runs": int(len(runs))}


def uniformity_test(counts: dict, priors: dict | None = None) -> dict:
    """Chi-square test of the winner distribution against uniform (or
    against measured priors when given)."""
    keys = sorted(counts)
    obs = np.array([counts[k] for k in keys], dtype=float)
    n = obs.sum()
    if priors:
        p = np.array([priors.get(k, 0) for k in keys], dtype=float)
        p = (p + 1e-9) / (p + 1e-9).sum()
    else:
        p = np.full(len(keys), 1 / len(keys))
    expected = n * p
    chi2 = float(np.sum((obs - expected) ** 2 / expected))
    dof = len(keys) - 1
    return {"chi2": chi2, "dof": dof,
            "p_value": float(stats.chi2.sf(chi2, dof)),
            "n": int(n), "observed": {k: int(counts[k]) for k in keys}}
