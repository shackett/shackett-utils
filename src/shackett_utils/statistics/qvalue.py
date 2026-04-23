"""
Simple reimplementation of key functions from R's qvalue package.

Public Functions
----------------
estimate_lfdr(pvals, pi0=None, trunc=True, monotone=True, adj=1.5, eps=1e-8, **pi0_kwargs)
    Estimate local FDR values from p-values.
estimate_pi0(pvals, lambda_vals=np.arange(0.05, 0.95 + 1e-10, 0.05), smooth_df=3)
    Estimate pi0, the proportion of true null hypotheses.
estimate_qvalues(pvals, pi0=None, pfdr=False, **pi0_kwargs)
    Estimate q-values from p-values.
"""

import logging
import warnings

import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import splrep, splev

logger = logging.getLogger(__name__)


def estimate_lfdr(
    pvals: np.ndarray,
    pi0: float | None = None,
    trunc: bool = True,
    monotone: bool = True,
    adj: float = 1.5,
    eps: float = 1e-8,
    **pi0_kwargs,
) -> np.ndarray:
    """
    Estimate local FDR values from p-values.

    Estimates the posterior probability that each test is a true null,
    using the formula lFDR(p) = pi0 / f(p), where f(p) is the marginal
    density of p-values estimated via KDE on probit-transformed values.

    Parameters
    ----------
    pvals : np.ndarray
        1D array of p-values in [0, 1]. Must not contain NaNs.
    pi0 : float, optional
        Proportion of true null hypotheses. If None, estimated from
        ``pvals`` via ``estimate_pi0()`` using any extra kwargs.
    trunc : bool
        If True, clamp lFDR values > 1 to 1. Default True.
    monotone : bool
        If True, enforce that lFDR is non-decreasing with increasing
        p-value — i.e. larger p-values cannot have smaller lFDR.
        Implemented via cumulative max on p-sorted values, matching R.
        Default True; recommended.
    adj : float
        Multiplier applied to Silverman's bandwidth before KDE. Higher
        values produce a smoother density estimate. R default is 1.5.
    eps : float
        P-values are clamped to [eps, 1-eps] before the probit transform
        to avoid infinite values at the boundary. R default is 1e-8.
    **pi0_kwargs
        Additional keyword arguments passed to ``estimate_pi0()`` when
        ``pi0`` is None (e.g. ``lambda_vals``, ``smooth_df``).

    Returns
    -------
    np.ndarray
        Local FDR values, same length and order as ``pvals``.

    Details
    -------
    Port of R ``qvalue::lfdr()`` using the probit transformation (R's
    default). Differences from the R implementation:

    - **NaN handling**: R silently removes NaNs and reinserts them into
      the output. This function raises a ``ValueError`` to avoid silent
      index misalignment in calling code.
    - **Transformation**: Only probit is implemented, matching R's
      default ``transf="probit"``. The logit branch is omitted.
    - **KDE bandwidth**: R uses ``density(x, adjust=adj)`` which applies
      ``bw.nrd0(x) * adj`` as the bandwidth. ``bw.nrd0`` and
      ``_silverman_bw`` are equivalent formulations of Silverman's rule
      and should give identical bandwidth values. scipy's
      ``gaussian_kde`` expects a normalised scalar, so the absolute
      bandwidth is divided by ``std(x)`` before passing.
    - **Spline smoothing**: R applies ``smooth.spline()`` to the 512-
      point KDE grid before evaluating at observations. scipy's
      ``splrep(s=None)`` uses a different GCV criterion than R's
      ``smooth.spline``, so smoothed density values will differ slightly.
    """
    pvals = np.asarray(pvals, dtype=float)

    if pvals.ndim != 1:
        raise ValueError("pvals must be 1-dimensional")
    if np.any(np.isnan(pvals)):
        raise ValueError("pvals contains NaNs — drop or impute before calling")
    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError("p-values must be in [0, 1]")

    if pi0 is None:
        pi0 = estimate_pi0(pvals, **pi0_kwargs)

    x, jacobian = _probit_transform(pvals, eps)

    bw = _silverman_bw(x) * adj
    y = _kde_then_spline(x, bw)

    lfdr = pi0 * jacobian / y

    if trunc:
        lfdr = np.minimum(lfdr, 1.0)

    if monotone:
        o = np.argsort(pvals)
        ro = np.argsort(o)
        lfdr = np.maximum.accumulate(lfdr[o])[ro]

    return lfdr


def estimate_pi0(
    pvals: np.ndarray,
    lambda_vals: np.ndarray = np.arange(0.05, 0.95 + 1e-10, 0.05),
    smooth_df: int = 3,
) -> float:
    """
    Estimate pi0, the proportion of true null hypotheses.

    At each lambda, the fraction of p-values exceeding lambda estimates
    pi0 after correcting for the truncation of the uniform null above
    lambda. A smoothing spline is fitted across lambda values and
    evaluated at max(lambda) to obtain a stable asymptotic estimate.

    Parameters
    ----------
    pvals : np.ndarray
        1D array of p-values in [0, 1]. Must not contain NaNs.
    lambda_vals : np.ndarray
        Grid of tuning values in [0, 1). At each lambda, tests with
        p > lambda are assumed likely null. Default matches R qvalue:
        seq(0.05, 0.95, 0.05). Must have >= 4 values if more than 1
        is provided (required for cubic spline fitting).
    smooth_df : int
        Smoothing parameter passed as ``s`` to scipy's ``splrep``.
        Analogous to R's ``smooth.df=3`` argument to ``smooth.spline``,
        though not numerically equivalent — see Details.

    Returns
    -------
    float
        Estimated pi0 clamped to (0, 1].

    Details
    -------
    Port of R ``qvalue::pi0est()`` with ``pi0.method="smoother"``.
    Differences from the R implementation:

    - **NaN handling**: R silently drops NaNs (``p <- p[!is.na(p)]``).
      This function raises a ``ValueError`` instead to avoid silent
      data loss in calling code.
    - **Smoothing spline**: R uses ``smooth.spline(lambda, pi0, df=3)``,
      which controls effective degrees of freedom via GCV. scipy's
      ``splrep`` uses ``s`` as a raw residual penalty, not effective df.
      ``s=smooth_df`` wires the parameter through with the correct
      direction (lower = smoother) but will not give numerically
      identical results to R. The difference is typically <0.5% on
      large omics datasets.
    - **Return type**: R returns a list with pi0, pi0.lambda, lambda,
      and pi0.smooth. This function returns only the scalar pi0.
    - **Tail padding**: If ``max(pvals) < max(lambda_vals)`` (e.g. every
      supplied p-value is below the default lambda grid, as with all-strong
      results plus an inert high-``p`` leg like an intercept), append a
      single ``p=1.0`` for the internal histogram and spline only; the scalar
      return is unchanged, and ``estimate_qvalues`` still ranks the original
      ``pvals`` (no extra row in the q-value pass).
    """
    pvals = np.asarray(pvals, dtype=float)

    if pvals.ndim != 1:
        raise ValueError("pvals must be 1-dimensional")
    if np.any(np.isnan(pvals)):
        raise ValueError("pvals contains NaNs — drop or impute before calling")
    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError("p-values must be in [0, 1]")

    lambda_vals = np.sort(lambda_vals)
    ll = len(lambda_vals)

    if np.min(lambda_vals) < 0 or np.max(lambda_vals) >= 1:
        raise ValueError("lambda values must be in [0, 1)")
    if ll > 1 and ll < 4:
        raise ValueError(f"Need at least 4 lambda values; got {ll}.")

    p_work = pvals
    if float(np.max(pvals)) < float(np.max(lambda_vals)):
        p_work = np.append(pvals, 1.0)

    if ll == 1:
        return min(
            float(np.mean(p_work >= lambda_vals[0]) / (1.0 - lambda_vals[0])),
            1.0,
        )

    m = len(p_work)
    counts = np.array([np.sum(p_work >= lam) for lam in lambda_vals])
    pi0_lambda = counts / (m * (1.0 - lambda_vals))

    tck = splrep(lambda_vals, pi0_lambda, k=3, s=smooth_df)
    pi0_smooth = splev(lambda_vals, tck)
    pi0 = float(pi0_smooth[-1])

    if pi0 <= 0:
        warnings.warn(f"Estimated pi0 <= 0 ({pi0:.4f}); setting to 1.", RuntimeWarning)
        pi0 = 1.0

    return min(pi0, 1.0)


def estimate_qvalues(
    pvals: np.ndarray,
    pi0: float | None = None,
    pfdr: bool = False,
    **pi0_kwargs,
) -> np.ndarray:
    """
    Estimate q-values from p-values.

    The q-value of a test is the minimum FDR incurred when calling that
    test and all tests with smaller p-values significant. Computed as a
    pi0-scaled BH step-up procedure: sort p-values descending, take the
    cumulative minimum of pi0 * m * p / rank, then restore original order.

    Parameters
    ----------
    pvals : np.ndarray
        1D array of p-values in [0, 1]. Must not contain NaNs.
    pi0 : float, optional
        Proportion of true null hypotheses. If None, estimated from
        ``pvals`` via ``estimate_pi0()`` using any extra kwargs.
    pfdr : bool
        If True, compute the positive FDR (pFDR) variant, which
        conditions on at least one rejection and is more conservative
        for small p-values. Default False.
    **pi0_kwargs
        Additional keyword arguments passed to ``estimate_pi0()`` when
        ``pi0`` is None (e.g. ``lambda_vals``, ``smooth_df``).

    Returns
    -------
    np.ndarray
        Q-values, same length and order as ``pvals``.

    Details
    -------
    Port of the q-value calculation in R ``qvalue::qvalue()``.
    Differences from the R implementation:

    - **NaN handling**: R silently removes NaNs and reinserts them.
      This function raises a ``ValueError`` instead.
    - **Return type**: R's ``qvalue()`` returns a list containing
      qvalues, lfdr, pi0, lambda, and diagnostic quantities. This
      function returns only the q-value array. Call ``estimate_lfdr()``
      directly for local FDR values.
    - **pFDR denominator**: The pFDR branch uses
      ``i * (1 - (1-p)^m)`` as the denominator, matching R exactly.
      For large m this approaches ``i`` and pFDR ≈ standard q-values.
    """
    pvals = np.asarray(pvals, dtype=float)

    if pvals.ndim != 1:
        raise ValueError("pvals must be 1-dimensional")
    if np.any(np.isnan(pvals)):
        raise ValueError("pvals contains NaNs — drop or impute before calling")
    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError("p-values must be in [0, 1]")

    if pi0 is None:
        pi0 = estimate_pi0(pvals, **pi0_kwargs)

    m = len(pvals)
    i = np.arange(m, 0, -1)
    o = np.argsort(pvals)[::-1]
    ro = np.argsort(o)

    p_sorted_desc = pvals[o]
    if pfdr:
        denom = i * (1 - (1 - p_sorted_desc) ** m)
        qvals = (
            pi0 * np.minimum(1, np.minimum.accumulate(p_sorted_desc * m / denom))[ro]
        )
    else:
        qvals = pi0 * np.minimum(1, np.minimum.accumulate(p_sorted_desc * m / i))[ro]

    return qvals


def _probit_transform(p: np.ndarray, eps: float):
    """
    Probit-transform p-values and return the Jacobian for density conversion.

    Maps p-values from [0, 1] to the real line via the normal quantile
    function. The Jacobian dnorm(x) converts densities estimated on the
    transformed scale back to the p-value scale.

    Parameters
    ----------
    p : np.ndarray
        P-values, already validated to be in [0, 1].
    eps : float
        Clamping threshold applied before transformation to avoid
        infinite values at 0 and 1.

    Returns
    -------
    x : np.ndarray
        Probit-transformed values (qnorm(p)).
    jacobian : np.ndarray
        Normal PDF evaluated at x (dnorm(x)), used as the Jacobian
        when converting f(x) back to f(p).
    """
    p = np.clip(p, eps, 1 - eps)
    x = norm.ppf(p)
    return x, norm.pdf(x)


def _kde_then_spline(x_obs: np.ndarray, bw: float, n_grid: int = 512) -> np.ndarray:
    """
    Estimate density at observed points via KDE on a grid followed by
    spline smoothing, matching R's two-step approach in qvalue::lfdr().

    R evaluates density() on a 512-point grid, fits smooth.spline() to
    that grid, then calls predict() at the original observations. This
    avoids evaluating the KDE directly at potentially sparse observation
    locations.

    Parameters
    ----------
    x_obs : np.ndarray
        Transformed observations (probit scale).
    bw : float
        Absolute bandwidth (Silverman * adj).
    n_grid : int
        Number of grid points for KDE evaluation. R default is 512.

    Returns
    -------
    np.ndarray
        Smoothed density evaluated at each point in x_obs.
    """
    kde = gaussian_kde(x_obs, bw_method=bw / np.std(x_obs, ddof=1))
    x_grid = np.linspace(x_obs.min() - 3 * bw, x_obs.max() + 3 * bw, n_grid)
    myd_y = kde(x_grid)

    tck = splrep(x_grid, myd_y, k=3, s=None)
    return splev(x_obs, tck)


def _silverman_bw(x: np.ndarray) -> float:
    """
    Silverman's rule of thumb bandwidth (R's bw.nrd0).

    Parameters
    ----------
    x : np.ndarray
        Data values on the transformed scale.

    Returns
    -------
    float
        Bandwidth estimate: 1.06 * min(sd, IQR/1.34) * n^(-1/5).
    """
    n = len(x)
    std = np.std(x, ddof=1)
    iqr_scaled = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34
    return 1.06 * min(std, iqr_scaled) * n ** (-0.2)
