from scipy import optimize as opt
from scipy.stats import norm, lognorm
import numpy as np
import lognorm3p, mle, gme, mde
import nlopt
import itertools


# def likelihood_log_normal(w, mu, sigma, X):
#     n = len(X)
#     m = len(w)
#     p = np.zeros(n)
#     for j in range(m):
#         p = p + w[j] * norm.pdf(X, loc=mu[j], scale=sigma[j])
#     print p
#     print np.log(p)
#     print np.sum(np.log(p))
#     print '-'
#     return np.sum(np.log(p))


# def em_normal(X, n_component=3, w0=None, m0=None, s0=None, max_iter=100, tol=1e-4, n_attempt=3):
#     n = len(X)
#     s = np.array(s0)
#     q = np.zeros((n_component, n))
#     F_max = -np.inf
#     for t in range(n_attempt):
#         F_opt = -np.inf
#         w = np.array(w0) + np.random.normal(scale=.01, size=n_component)
#         w = np.abs(w) / np.sum(np.abs(w))
#         m = np.array(m0) + np.random.standard_normal(n_component)
#         s = np.fmax(.01, np.array(s0) + np.random.normal(scale=.01, size=n_component))
#         print w
#         print m
#         print s
#         print '---'
#         for iter in range(max_iter):
#             for j in range(n_component):
#                 q[j, :] = w[j] * norm.pdf(X, loc=m[j], scale=s[j])
#             z = q.sum(axis=0)
#             idx = z > 0
#             q[:, idx] /= z[np.newaxis, idx]
#             w = q.sum(axis=1)
#             w /= n
#             idx = w != 0
#             w = w[idx]
#             m = m[idx]
#             s = s[idx]
#             print w
#             print m
#             print s
#             print '--------'
#             q = q[idx, :]
#             n_component = sum(idx)
#             for j in range(n_component):
#                 m[j] = np.sum(q[j, :] * X) / np.sum(q[j, :])
#                 s[j] = (np.sum(q[j, :] * (X - m[j])**2) / np.sum(q[j, :]))**.5
#             F = F_opt
#             F_opt = likelihood_log_normal(w, m, s, X)
#             if F_opt - F < tol:
#                 break
#         if F_opt > F_max:
#             w_max = w
#             m_max = m
#             s_max = s
#     w = w_max
#     m = m_max
#     s = s_max
#     g = m - 3 * s
# #    for j in range(n_component):
# #        idx = X > g[j]
# #        m[j] = mle.likelihood_mu(g[j], X[idx], q[j, idx])
# #        s[j] = mle.likelihood_sigma(g[j], X[idx], q[j, idx])
#     return w, m, s, F_opt - F


def cdf(x, w, gamma, mu, sigma):
    x = np.asarray(x)
    #f = np.zeros(x.shape)
    t = np.exp(mu)
    gamma = np.asarray(gamma)
    sigma = np.asarray(sigma)
    f = lognorm.cdf(x, sigma[:, np.newaxis], loc=gamma[:, np.newaxis], scale=t[:, np.newaxis])
    #for j in range(m):
        #f = f + w[j] * lognorm.cdf(x, sigma[j], loc=gamma[j], scale=t[j])
        #lognorm3p.cdf(x, gamma=gamma[j], mu=mu[j], sigma=sigma[j])
    return np.dot(w, f)


def pdf(x, w, gamma, mu, sigma):
    x = np.asarray(x)
    #f = np.zeros(x.shape)
    t = np.exp(mu)
    gamma = np.asarray(gamma)
    sigma = np.asarray(sigma)
    f = lognorm.pdf(x, sigma[:, np.newaxis], loc=gamma[:, np.newaxis], scale=t[:, np.newaxis])
    #for j in range(m):
        #f = f + w[j] * lognorm.cdf(x, sigma[j], loc=gamma[j], scale=t[j])
        #lognorm3p.cdf(x, gamma=gamma[j], mu=mu[j], sigma=sigma[j])
    return np.dot(w, f)


def likelihood_log(w, gamma, mu, sigma, X, suppress=0.):
    x = np.asarray(X)
    t = np.exp(mu)
    gamma = np.asarray(gamma)
    sigma = np.asarray(sigma)
    p = lognorm.pdf(x, sigma[:, np.newaxis], loc=gamma[:, np.newaxis], scale=t[:, np.newaxis])
    t = np.dot(w, p)
    #t[t < suppress] = suppress
    #for j in range(m):
        #p = p + w[j] * lognorm.cdf(x, sigma[j], loc=gamma[j], scale=t[j])
    return np.sum(np.log(t[t > 0])) / np.sum(t > 0)


def em(X, n_component=3, w0=None, g0=None, m0=None, s0=None,
       estimator=lambda X, g: mle.estimate(X, g, name='standard'), max_iter=100, tol=1e-4, suppress=0., local=None):
    n = len(X)
    w = np.array(w0)
    g = np.array(g0)
    m = np.array(m0)
    s = np.array(s0)
    q = np.zeros((n_component, n))
    F_opt = -np.inf
    for iter in range(max_iter):
        w0, g0, m0, s0 = w, g, m, s
        q = w[:, np.newaxis] * lognorm.pdf(X, s[:, np.newaxis], loc=g[:, np.newaxis], scale=np.exp(m)[:, np.newaxis])
        z = q.sum(axis=0)
        idx = z > 0
        q[:, idx] /= z[np.newaxis, idx]
        w = q.sum(axis=1)
        w /= np.sum(w)
        idx = w != 0
        w = w[idx]
        g = g[idx]
        m = m[idx]
        s = s[idx]
        q = q[idx, :]
        n_component = sum(idx)
        for j in range(n_component):
            z = q[j, :]
            #z[z < suppress] = 0
            gt, mt, st, r = estimator(X[z > suppress], z[z > suppress])
            if r:
                if local is not None:
                    step = lambda alpha: - mle.likelihood_log(g[j] + alpha * (gt - g[j]),
                                                              m[j] + alpha * (mt - m[j]),
                                                              s[j] + alpha * (st - s[j]),
                                                              X[z > suppress], z[z > suppress])
                    o = opt.minimize_scalar(step, method='bounded', bounds=[0, 1])
                    alpha = o.x
                    gt, mt, st = g[j] + alpha * (gt - g[j]), m[j] + alpha * (mt - m[j]), s[j] + alpha * (st - s[j])
                if mle.likelihood_log(gt, mt, st, X[z > suppress], z[z > suppress]) < \
                mle.likelihood_log(g[j], m[j], s[j], X[z > suppress], z[z > suppress]):
                    gt, mt, st = g[j], m[j], s[j]
                g[j] = gt
                m[j] = mt
                s[j] = st
        F = F_opt
        F_opt = likelihood_log(w, g, m, s, X)
        if 0 <= F_opt - F < tol:
            w0, g0, m0, s0 = w, g, m, s
            break
        if F_opt - F < 0:
            break
    return w0, g0, m0, s0, F_opt - F


def dist_fun(w, gamma, mu, sigma, X, name='Cramer-von Mises'):
    n = len(X)
    X = np.sort(X)
    g = np.ones(n) / n
    t = np.cumsum(g)
    r = cdf(X, w, gamma, mu, sigma)
    if name == 'Kolmogorov-Smirnov':
        t = np.insert(t, 0, 0.)
        return max(np.max(np.abs(r - t[:-1])), np.max(np.abs(r - t[1:])))
    if name == 'Kuiper':
        t = np.insert(t, 0, 0.)
        return max(np.max(t[1:] - r), 0.) - np.min(np.min(t[:-1] - r), 0)
    if name == 'Cramer-von Mises':
        t[1:] = t[1:] + t[:-1]
        return (1. / 3 + np.sum(g * r * (r - t))) * n
    if name == 'Anderson-Darling':
        t[1:] = t[1:] + t[:-1]
        return (-1 - np.sum(g * t * np.log(r)) + np.sum(g * (t - 2) * np.log(1 - r))) * n
    if name == 'Watson':
        return dist_fun(w, gamma, mu, sigma, X, name='Cramer-von Mises') - n * (np.sum(g * r) - .5) ** 2

    
def dist_fun_allinone(theta, X, name='Cramer-von Mises'):
    n = len(X)
    m = (len(theta) + 1) / 4
    #X = np.sort(X)
    #g = np.ones(n) / n
    #t = np.cumsum(g)
    w = theta[:(m - 1)]
    w = np.append(w, [1 - w.sum()])
    gamma = theta[(m - 1):(2 * m - 1)]
    mu = theta[(2 * m - 1):(3 * m - 1)]
    sigma = theta[(3 * m - 1):(4 * m - 1)]
    s = dist_fun(w, gamma, mu, sigma, X, name=name)
    return s
    #r = cdf(X, w, gamma, mu, sigma)
    #if name == 'Kolmogorov-Smirnov':
        #t = np.insert(t, 0, 0.)
        #return max(np.max(np.abs(r - t[:-1])), np.max(np.abs(r - t[1:])))
    #if name == 'Kuiper':
        #t = np.insert(t, 0, 0.)
        #return max(np.max(t[1:] - r), 0.) - np.min(np.min(t[:-1] - r), 0)
    #if name == 'Cramer-von Mises':
        #t[1:] = t[1:] + t[:-1]
        #return (1. / 3 + np.sum(g * r * (r - t))) * n
    #if name == 'Anderson-Darling':
        #t[1:] = t[1:] + t[:-1]
        #return (-1 - np.sum(g * t * np.log(r)) + np.sum(g * (t - 2) * np.log(1 - r))) * n
    #if name == 'Watson':
        #return dist_fun(theta, X, name='Cramer-von Mises') - n * (np.sum(g * r) - .5) ** 2
    

def min_dist(X, dist='Cramer-von Mises', n_component = 3, w0=None, g0=None, m0=None, s0=None,
             tol=1e-4, max_iter=1000, method='PRAXIS'):
    m = n_component
    if method == 'PRAXIS':
        local_opter = nlopt.opt(nlopt.LN_PRAXIS, 4 * m - 1)
        local_opter.set_ftol_abs(tol)
        local_opter.set_maxeval(max_iter)
        opter = nlopt.opt(nlopt.AUGLAG, 4 * m - 1)
        opter.set_local_optimizer(local_opter)
    if method == 'COBYLA':
        opter = nlopt.opt(nlopt.LN_COBYLA, 4 * m - 1)
    opter.set_min_objective(lambda theta, grad: dist_fun_allinone(theta, X, name=dist))
    lb = np.array([0.] * (m - 1) + [-np.inf] * (2 * m) + [0.] * m)
    ub = np.array([1.] * (m - 1) + [np.inf] * (3 * m))
    opter.set_lower_bounds(lb)
    opter.set_upper_bounds(ub)
    opter.add_inequality_constraint(lambda theta, grad: np.sum(theta[:(m - 1)]) - 1)
    theta0 = np.append(np.array(w0[:-1]), np.append(np.array(g0), np.append(np.array(m0), np.array(s0))))
    opter.set_ftol_abs(tol)
    opter.set_maxeval(max_iter)
    try:
        theta = opter.optimize(theta0)
    except (nlopt.RoundoffLimited, ValueError):
        theta = np.array([np.nan] * (4 * m - 1))
    w = theta[:(m - 1)]
    w = np.append(w, [1 - w.sum()])
    gamma = theta[(m - 1):(2 * m - 1)]
    mu = theta[(2 * m - 1):(3 * m - 1)]
    sigma = theta[(3 * m - 1):(4 * m - 1)]
    r = opter.last_optimize_result()
    return w, gamma, mu, sigma, r# in [nlopt.XTOL_REACHED, nlopt.FTOL_REACHED]


def em_net(X, g_net, m_net, s_net, max_iter=100, tol=1e-4):
    n = len(X)
    g_net = np.array(g_net)
    m_net = np.array(m_net)
    s_net = np.array(s_net)
    l = lognorm.pdf(X, 
                    s_net[:, np.newaxis], 
                    loc=g_net[:, np.newaxis], 
                    scale=np.exp(m_net[:, np.newaxis]))
    w = np.ones(len(g_net)) / (len(g_net))
    L = lambda p: np.sum(np.log(np.sum(w[:, np.newaxis] * l, axis=0)))
    L_opt = -np.inf
    for iter in range(max_iter):
        q = w[:, np.newaxis] * l
        z = q.sum(axis=0)
        idx = z > 0
        q[:, idx] /= z[np.newaxis, idx]
        w = q.sum(axis=1)
        w /= n
        L_old = L_opt
        L_opt = L(w)
        if L_opt - L_old < tol:
            break
    #p = np.array(list(itertools.product(g_net, m_net, s_net)))
    return w.flatten(), g_net, m_net, s_net, L_opt - L_old#p[:, 0], p[:, 1], p[:, 2]

