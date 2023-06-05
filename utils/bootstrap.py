import warnings
import numpy as np
from sklearn.utils import resample
from scipy.special import ndtr, ndtri

def _percentile_of_score(a, score):
    """Vectorized, simplified `scipy.stats.percentileofscore`.

    Unlike `stats.percentileofscore`, the percentile returned is a fraction
    in [0, 1].
    """
    B = a.shape[1]
    return (a < score.T).sum(axis=1) / B


def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    # the difference between _percentile_along_axis and np.percentile is that
    # np.percentile gets _all_ the qs for each axis slice, whereas
    # _percentile_along_axis gets the q corresponding with each axis slice
    shape = theta_hat_b.shape[:-1]
    alpha = np.broadcast_to(alpha, shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            # e.g. when bootstrap distribution has only one unique element
            msg = ("The bootstrap distribution is degenerate; the "
                   "confidence interval is not defined.")
            warnings.warn(msg)
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]  # return scalar instead of 0d array


def _bca_interval(X, y, func, alpha, theta_hat_b, avoid_zero):
    """Bias-corrected and accelerated interval."""
    # closely follows [2] "BCa Bootstrap CIs"

    # calculate z0_hat
    theta_hat = func(X, y)
    percentile = _percentile_of_score(theta_hat_b, theta_hat)
    z0_hat = ndtri(percentile)

    # calculate a_hat
    nsubj, nfeat = X.shape
    idx = np.ones((nsubj,), dtype=bool)
    theta_hat_i = np.empty((nfeat, nsubj))
    for subj in range(0, len(idx)):
        idx[subj] = False
        X_knife = X[idx, :]
        y_knife = y[idx]
        theta_hat_i[:, subj] = func(X_knife, y_knife)
        idx[subj] = True
    
    theta_hat_dot = np.mean(theta_hat_i, axis=1, keepdims=True)
    num = ((theta_hat_dot - theta_hat_i)**3).sum(axis=1)
    den = 6*((theta_hat_dot - theta_hat_i)**2).sum(axis=1)**(3/2)
    if avoid_zero:
        den += 1e-10
    a_hat = num / den

    # calculate alpha_1, alpha_2
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1/(1 - a_hat*num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2/(1 - a_hat*num2))
    return alpha_1, alpha_2


def bootstrap(X, y, func, nboots, confidence_level=0.95,
              method='bca', avoid_zero=True, random_state=None):

    np.random.seed(random_state)
    nsubj, nfeat = X.shape
    theta_hat_b = np.empty((nfeat, nboots))
    data_idx = np.arange(0, nsubj, 1, dtype=int)
    
    for boot in range(0, nboots):
        resampled = resample(data_idx, replace=True, stratify=y) 
        X_boot = X[resampled, :]
        y_boot = y[resampled]
        theta_hat_b[:, boot] = func(X_boot, y_boot)

    # Calculate percentile interval
    alpha = (1 - confidence_level)/2
    if method == 'bca':
        interval = _bca_interval(X, y, func, alpha=alpha, 
                                 theta_hat_b=theta_hat_b, avoid_zero=avoid_zero)
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = func(X_boot, y_boot)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l
        
    resboot = {'LowerCI': ci_l,
               'UpperCI': ci_u,
               'Coefficients': theta_hat_b}
        
    return resboot