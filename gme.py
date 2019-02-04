from scipy import optimize as opt
from scipy.stats import norm
from scipy import special
import numpy as np


def moments_m(X, r=1, g=None):
    n = len(X)
    if g is None:
        g = np.ones(n)
    V1 = np.sum(g)
    if r == 1:
        return np.sum(g * X) / V1
    else:
        return np.sum(g * (X - moments_m(X, r=1, g=g))**r) / V1


def moments_sigma(o, X, g=None):
    return np.log(o)**.5


def moments_mu(o, X, g=None):
    n = len(X)
    if g is None:
        g = np.ones(n)
    V1 = np.sum(g)
    V2 = np.sum(g**2)
    M2 = (V1**2 / (V1**2 - V2)) * moments_m(X, r = 2, g = g)
    return np.log(M2) / 2 - np.log(o * (o - 1)) / 2


def moments_gamma(o, X, g=None):
    n = len(X)
    if g is None:
        g = np.ones(n)
    V1 = np.sum(g)
    V2 = np.sum(g**2)
    M1 = moments_m(X, r = 1, g = g)
    M2 = (V1**2 / (V1**2 - V2)) * moments_m(X, r = 2, g = g)
    return M1 - (M2 / (o - 1))**.5
    
    
# def moments_gme(X, g = None):
#     n = len(X)
#     if g is None:
#         g = np.ones(n)
#     V1 = np.sum(g)
#     V2 = np.sum(g**2)
#     V3 = np.sum(g**3)
#     M2 = (V1**2 / (V1**2 - V2)) * moments_m(X, r = 2, g = g)
#     M3 = (V1**3 / (V1**3 - 3 * V1 * V2 + 2 * V3)) * moments_m(X, r = 3, g = g)
#     a3 = M3 / M2**1.5
#     v = ((a3 + (4 + a3**2)**0.5)/2)**(1./3) - ((-a3 + (4 + a3**2)**0.5)/2)**(1./3)
#     o = 1 + v**2
#     return (moments_gamma(o, X, g), moments_mu(o, X, g), moments_sigma(o, X, g))


# def moments_gme1(X, r = 1, g = None):
#     n = len(X)
#     if g is None:
#         g = np.ones(n)
#     idx = np.argsort(X)
#     X = X[idx]
#     g = g[idx]
#     g = g / np.sum(g)
#     s = np.cumsum(g)
#     while s[r - 1] == 0:
#         r = r + 1
#     kr = norm.ppf(s[r - 1] / (1 + g[r - 1]))
#     V1 = np.sum(g)
#     V2 = np.sum(g**2)
#     M1 = moments_m(X, r = 1, g = g)
#     M2 = (V1**2 / (V1**2 - V2)) * moments_m(X, r = 2, g = g)
#     J = lambda o: o * (o - 1) / (o**.5 - np.exp(np.log(o)**.5 * kr)) - M2 / (M1 - X[r - 1])**2
#     o = opt.brenth(J, 1 + 1e-8, 100)   #TODO: bounds
#     return (moments_gamma(o, X, g), moments_mu(o, X, g), moments_sigma(o, X, g))


def estimate(X, g=None, name='standard', r=1):
    n = len(X)
    if g is None:
        g = np.ones(n)
    gamma = np.nan
    mu = np.nan
    sigma = np.nan
    res = False
    if name == 'standard':
        V1 = np.sum(g)
        V2 = np.sum(g**2)
        V3 = np.sum(g**3)
        M2 = (V1**2 / (V1**2 - V2)) * moments_m(X, r = 2, g = g)
        M3 = (V1**3 / (V1**3 - 3 * V1 * V2 + 2 * V3)) * moments_m(X, r = 3, g = g)
        a3 = M3 / M2**1.5
        v = ((a3 + (4 + a3**2)**0.5)/2)**(1./3) - ((-a3 + (4 + a3**2)**0.5)/2)**(1./3)
        o = 1 + v**2
        gamma = moments_gamma(o, X, g)
        mu = moments_mu(o, X, g)
        sigma = moments_sigma(o, X, g)
        res = True
    if name == 'modified':
        idx = np.argsort(X)
        X = X[idx]
        g = g[idx]
        g = g / np.sum(g)
        s = np.cumsum(g)
        while s[r - 1] == 0:
            r += 1
        kr = norm.ppf(s[r - 1] / (1 + g[r - 1]))
        V1 = np.sum(g)
        V2 = np.sum(g**2)
        M1 = moments_m(X, r = 1, g = g)
        M2 = (V1**2 / (V1**2 - V2)) * moments_m(X, r = 2, g = g)
        J = lambda o: o * (o - 1) / (o**.5 - np.exp(np.log(o)**.5 * kr)) - M2 / (M1 - X[r - 1])**2

        a = 1 + 1e-6
        while a > 1 + 1e-100:
            if J(a) < 0:
                break
            else:
                a = (a + 1.) / 2.
        b = 2 * a - 1
        while b < 1e+100:
            if J(b) > 0:
                break
            else:
                b = 2 * b - a
        if J(a) > 0 or J(b) < 0:
            gamma = np.nan
            mu = np.nan
            sigma = np.nan
            res = False
        else:
            o = opt.brenth(J, a, b)
            gamma = moments_gamma(o, X, g)
            mu = moments_mu(o, X, g)
            sigma = moments_sigma(o, X, g)
            res = True
    if name == 'L-moments':
        idx = np.argsort(X)
        X = X[idx]
        g = g[idx]
        idx = g > 0
        X = X[idx]
        g = g[idx]
        m = len(g)
        if m == 1:
            return X[0], X[0], 1
        if m == 0:
            return 0, 0, 1
        g = g / np.sum(g) * m
        l1 = np.sum(g * X) / m #np.sum(g) = m
        p = np.cumsum(g)
        q = np.cumsum(g[::-1])[::-1]
        t = p - q
        l2 = 1. / (m * (m - 1)) * np.sum(X * g * t)
        r = np.cumsum(g * p)
        s = np.cumsum((g * q)[::-1])[::-1]
        p = np.insert(p, 0, [0., 0.])
        r = np.insert(r, 0, [0., 0.])
        q = np.append(q, [0., 0.])
        s = np.append(s, [0., 0.])
        g = np.append(np.insert(g, 0, 0.), 0.)
        l3 = 2. * np.sum(((X * 1.) / (m - 1) / (m - 2)) * (g[1:-1] / m) * (t * (p[:-2] - q[2:]) - r[:-2] - s[2:] - g[:-2] * q[1:-1] - g[2:] * p[1:-1]))
        t3 = l3 / l2
        z = (8. / 3.)**.5 * norm.ppf((1. + t3) / 2.)
        sigma = 0.999281 * z - 0.006118 * z**3 + 0.000127 * z**5
        if sigma > 0:
            mu = np.log(l2 / (special.erf(sigma / 2.))) - sigma**2 / 2.
            gamma = l1 - np.exp(mu + sigma**2 / 2.)
            res = True
        else:
            mu = np.nan
            gamma = np.nan
            res = False
    return gamma, mu, sigma, res
