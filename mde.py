from scipy import optimize as opt
from scipy.stats import norm
import nlopt
import numpy as np
import lognorm3p

# def distance_cvm(gamma, mu, sigma, X, g = None):
#     n = len(X)
#     if g is None:
#         g = np.ones(n)
#     idx = np.argsort(X)
#     X = X[idx]
#     g = g[idx]
#     g = g / np.sum(g)
#     t = np.cumsum(g)
#     t[1:] = t[1:] + t[:-1]
#     r = lognorm3p.cdf(X, gamma=gamma, mu=mu, sigma=sigma)
#     return 1. / 3 + np.sum(g * r * (r - t))


# def distance_ad(gamma, mu, sigma, X, g = None):
#     n = len(X)
#     if g is None:
#         g = np.ones(n)
#     idx = np.argsort(X)
#     X = X[idx]
#     g = g[idx]
#     g = g / np.sum(g)
#     t = np.cumsum(g)
#     t[1:] = t[1:] + t[:-1]
#     r = lognorm3p.cdf(X, gamma=gamma, mu=mu, sigma=sigma)
#     return -1 - np.sum(g * t * np.log(r)) + np.sum(g * (t - 2) * np.log(1 - r))


# def distance_ks(gamma, mu, sigma, X, g = None):
#     n = len(X)
#     if g is None:
#         g = np.ones(n)
#     idx = np.argsort(X)
#     X = X[idx]
#     g = g[idx]
#     g = g / np.sum(g)
#     t = np.insert(np.cumsum(g), 0, 0.)
#     r = lognorm3p.cdf(X, gamma=gamma, mu=mu, sigma=sigma)
#     return max(np.max(np.abs(r - t[:-1])), np.max(np.abs(r - t[1:])))


def dist(gamma, mu, sigma, X, g = None, name = 'Cramer-von Mises'):
    n = len(X)
    if g is None:
        g = np.ones(n)
    idx = np.argsort(X)
    X = X[idx]
    g = g[idx]
    n = len(X)
    g = g / np.sum(g)
    t = np.cumsum(g)
    r = lognorm3p.cdf(X, gamma=gamma, mu=mu, sigma=sigma)
    if name == 'Kolmogorov-Smirnov':
        t = np.insert(np.cumsum(g), 0, 0.)
        return max(np.max(np.abs(r - t[:-1])), np.max(np.abs(r - t[1:])))
    if name == 'Kuiper':
        t = np.insert(t, 0, 0.)
        return max(np.max(t[1:] - r), 0.) - np.min(np.min(t[:-1] - r), 0)
    if name == 'Cramer-von Mises':
        t[1:] = t[1:] + t[:-1]
        return (1. / 3 + np.sum(g * r * (r - t))) * n
    if name == 'Anderson-Darling':
        t[1:] = t[1:] + t[:-1]
        r = np.clip(r, 1e-7, 1 - 1e-7) #numerical stability
        return (-1 - np.sum(g * t * np.log(r)) + np.sum(g * (t - 2) * np.log(1 - r)))*n
    if name == 'Watson':
        return dist(gamma, mu, sigma, X, g, name='Cramer-von Mises') - n * (np.sum(g * r) - .5)**2
    # if name == 'Frozini':
    #     r = np.insert(r, 0, 0.)
    #     r = np.append(r, [1.])
    #     t = np.insert(t, 0, 0.)
    #     p = r[1:] - t
    #     q = r[:-1] - t
    #     return n**.5 * np.sum(p * np.abs(p) - q * np.abs(q)) / 2.


def estimate(X, g=None, distance='Cramer-von Mises', x0=(0., 0., 1.), method='Nelder-Mead', bounds=None,
             step_size=None, stopval=1e-4, max_iter=1000):
    d = lambda x: dist(x[0], x[1], x[2], X, g=g, name=distance)
    if method in ['Nelder-Mead', 'Powell', 'L-BFGS-B']:
        z = opt.minimize(d, x0, method=method, bounds=bounds)
        return z.x[0], z.x[1], z.x[2], z.success
    if method in ['BOBYQA', 'PRAXIS'] or isinstance(method, (int, long)):
        m = method#np.nan
        if method == 'BOBYQA':
            m = nlopt.LN_BOBYQA
        if method == 'PRAXIS':
            m = nlopt.LN_PRAXIS
        opter = nlopt.opt(m, 3)
        opter.set_min_objective(lambda x, grad: d(x))
        if bounds is not None:
            lb = np.array([-np.inf, -np.inf, -np.inf])
            ub = np.array([np.inf, np.inf, np.inf])
            for i in [0, 1, 2]:
                if bounds[i][0] is not None:
                    lb[i] = bounds[i][0]
                if bounds[i][1] is not None:
                    ub[i] = bounds[i][1]
            opter.set_lower_bounds(lb)
            opter.set_upper_bounds(ub)
        if step_size is not None:
            #z = [0, 0, 1]
            #opter.get_initial_step(x0, z)
            #print z
            opter.set_initial_step(step_size)
        opter.set_ftol_rel(stopval)
        opter.set_maxeval(max_iter)
        try:
            x = opter.optimize(np.array(x0))
        except (nlopt.RoundoffLimited, ValueError):
            x = [np.nan, np.nan, np.nan]
        r = opter.last_optimize_result()
        return x[0], x[1], x[2], r in [nlopt.FTOL_REACHED, nlopt.XTOL_REACHED]#, nlopt.SUCCESS]

