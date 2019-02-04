from scipy import optimize as opt
from scipy.stats import norm
import numpy as np


def pdf(x, gamma=0., mu=0., sigma=1.):
    x = np.asarray(x)
    z = np.zeros(x.shape)
    z[x > gamma] = norm.pdf(np.log(x[x > gamma] - gamma), loc = mu, scale = sigma) / (x[x > gamma] - gamma)
    return z + 0.0


def cdf(x, gamma=0., mu=0., sigma=1.):
    x = np.asarray(x)
    z = np.zeros(x.shape)
    z[x > gamma] = norm.cdf(np.log(x[x > gamma] - gamma), loc = mu, scale = sigma)
    return z + 0.0


def ppf(p, gamma=0., mu=0., sigma=1.):
    return gamma + np.exp(norm.ppf(p, loc = mu, scale = sigma)) + 0.0

