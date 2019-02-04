from scipy import optimize as opt
from scipy.stats import norm
import numpy as np


def likelihood_log(gamma, mu, sigma, X, g=None):
    n = len(X)
    if g is None:
        g = np.ones(n)
    idx = g > 0
    X = X[idx]
    g = g[idx]
    if np.all(X > gamma):
        return np.sum(g * (norm.logpdf(np.log(X - gamma), loc=mu, scale=sigma) - np.log(X - gamma)))
    else:
        return - np.inf


def likelihood_mu(gamma, X, g=None):
    n = len(X)
    if g is None:
        g = np.ones(n)
    return np.sum(g * np.log(X - gamma)) / np.sum(g)


def likelihood_sigma(gamma, X, g = None):
    n = len(X)
    if g is None:
        g = np.ones(n)
    return (np.sum(g * (np.log(X - gamma) - likelihood_mu(gamma, X, g))**2) / np.sum(g))**0.5


def likelihood_equation(gamma, X, g=None, method='standard', r = 1):
    n = len(X)
    if g is None:
        g = np.ones(n)
    if method == 'standard':
        return np.sum(g / (X - gamma) * (1 + (np.log(X - gamma) - likelihood_mu(gamma, X, g)) / likelihood_sigma(gamma, X, g)**2))
    if method == 'modified':
        idx = np.argsort(X)
        X = X[idx]
        g = g[idx]
        g = g / np.sum(g)
        s = np.cumsum(g)
        while s[r - 1] == 0:
            r += 1
        kr = norm.ppf(s[r - 1] / (1 + g[r - 1]))  # when g = 1/n, it's ordinary r/(n + 1); it comes from beta-distribution weight fix
        return np.log(X[r - 1] - gamma) - likelihood_mu(gamma, X, g) - kr * likelihood_sigma(gamma, X, g)


def estimate(X, g=None, name='standard', r=1):
    n = len(X)
    if g is None:
        g = np.ones(n)
    idx = g == 0.
    g = g[np.logical_not(idx)]
    X = X[np.logical_not(idx)]
    l = lambda gamma: likelihood_equation(gamma, X, g=g, method=name, r=r)
    a = np.min(X) - 1e-5
    u = a
    while np.abs(u - np.min(X)) > 1e-9: # 1e-100
        if l(u) < 0:
            a = u
            break
        else:
            u = (u + np.min(X)) / 2.
    while u > -10000: # -1e+100
        if l(u) < 0:
            a = u
            break
        else:
            u = 2 * u - np.min(X)
    if l(a) > 0:
        return np.nan, np.nan, np.nan, False
    b = a - 1e-6
    while b > -10000: # -1e+100
        if l(b) > 0:
            break
        else:
            b = 2 * b - a
    if l(b) < 0:
        return np.nan, np.nan, np.nan, False
    gamma = opt.brenth(l, a, b)
    mu = likelihood_mu(gamma, X, g)
    sigma = likelihood_sigma(gamma, X, g)
    return gamma, mu, sigma, True

