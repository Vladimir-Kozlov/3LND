import lognorm3p, mixture as mix
from scipy.integrate import quad
import numpy as np
import math

def ds_theoretic(x, order=1, p=None, lb=-np.inf):
    if x <= lb:
        return 0
    else:
        return quad(lambda y: (x - y)**(order - 1) * p(y), lb, x)[0] / math.factorial(order - 1)


def ds_empiric(x, order=1, X=None):
    X = np.asarray(X)
    return np.mean((x >= X) * (x - X)**(order - 1)) / (math.factorial(order - 1))


def pdf_sum(x, w1, g1, m1, s1, w2, g2, m2, s2):
    a = np.amin(np.asarray(g1))
    b = np.amin(np.asarray(g2))
    if x <= a + b:
        return 0.
    else:
        return quad(lambda y: mix.pdf(y, w1, g1, m1, s1) * mix.pdf(x - y, w2, g2, m2, s2), a, x - b)[0]


def cdf_sum(x, w1, g1, m1, s1, w2, g2, m2, s2):
    a = np.amin(np.asarray(g1))
    b = np.amin(np.asarray(g2))
    if x <= a + b:
        return 0.
    else:
        return quad(lambda y: mix.pdf(y, w1, g1, m1, s1) * mix.cdf(x - y, w2, g2, m2, s2), a, x - b)[0]


def test_statistic(X, Y, order=1):
    N = len(X)
    M = len(Y)
    Z = np.unique(np.concatenate((X, Y)))
    return np.array([ds_empiric(z, order, Y) - ds_empiric(z, order, X)
                     for z in Z]).max() * ((N * M * 1.) / (N + M))**.5


# def lim_statistic(X, Y, order=1):
#     N = len(X)
#     M = len(Y)
#     X1 = np.random.choice(X, size=len(X), replace=True)
#     Y1 = np.random.choice(Y, size=len(Y), replace=True)
#     Z = np.unique(np.concatenate((X, Y)))
#     return np.array([(ds_empiric(z, order, Y1) - ds_empiric(z, order, Y)) - (ds_empiric(z, order, X1) - ds_empiric(z, order, X))
#                      for z in Z]).max() * ((N * M * 1.) / (N + M))**.5


def lim_statistic(X, Y, order=1):
    Z = np.concatenate((X, Y))
    N = len(X)
    M = len(Y)
    X = np.random.choice(Z, size=N, replace=True)
    Y = np.random.choice(Z, size=M, replace=True)
    return np.array([ds_empiric(z, order, Y) -  ds_empiric(z, order, X) for z in np.unique(Z)]).max() * ((N * M * 1.) / (N + M))**.5


def p_value(X, Y, order=1, r=100):
    N = len(X)
    M = len(Y)
    Z = np.concatenate((X, Y))
    W = np.unique(Z)
    s = np.array([ds_empiric(z, order, Y) - ds_empiric(z, order, X) for z in W]).max() #* ((N * M * 1.) / (N + M))**.5
    k = 0
    for i in range(r):
        X0 = np.random.choice(Z, size=N, replace=True)
        Y0 = np.random.choice(Z, size=M, replace=True)
        t = np.array([ds_empiric(z, order, Y0) -  ds_empiric(z, order, X0) for z in W]).max() #* ((N * M * 1.) / (N + M))**.5
        k += (t > s)
    return (k + 0.) / r #np.mean([lim_statistic(X, Y, order) > s for i in range(r)])

