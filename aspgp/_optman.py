"""
A python implementation of:

    A feasible method for optimization with orthogonality constraints,
    Zaiwen, Wn, Wotao, Yin

Author:
    Ilias Bilionis

Date:
    4/18/2015

"""


import numpy as np


def opt_stiefel_gbb(fun, X, args=(),
                    disp=False,
                    mxitr=1000,
                    xtol=1e-6,
                    gtol=1e-6,
                    ftol=1e-12,
                    rho=1e-4,
                    eta=0.1,
                    gamma=0.85,
                    tau_max=1e-3,
                    STEPS=1e-10,
                    nt=5,
                    projG=1,
                    iscomplex=0):
    """
    Write this.
    """
    n, k = X.shape
    invH = True
    if k < n / 2:
        invH = False
        eye2k = np.eye(2 * k)

    F, G = fun(X, *args)
    nfe = 1
    GX = np.dot(G.T, X)

    if invH:
        GXT = np.dot(G, X.T)
        H = 0.5 * (GXT - GXT.T)
        RX = np.dot(H, X)
    else:
        if projG == 1:
            U = np.hstack([G, X])
            V = np.hstack([X, -G])
            VU = np.dot(V.T, U)
        elif projG == 2:
            GB = G - 0.5 * np.dot(X, np.dot(X.T, G))
            U = np.hstack([GB, X])
            V = np.hstack([X, -GB])
            VU = np.dot(V.T, U)
        VX = np.dot(V.T, X)
    dtX = G - np.dot(X, GX)
    nrmG = np.linalg.norm(dtX, 'fro')

    Q = 1
    Cval = F
    tau = tau_max
    success = False

    if disp:
        print 'Gradient Method with Line Search'.center(80)
        print '%4s %8s %8s %10s %10s' %('Iter', 'tau', 'F(X)', 'nrmG', 'XDiff')
        print '%4d \t %3.2e \t %3.2e \t %5d \t %5d \t %6d' % (0, 0, F, 0, 0, 0)

    for itr in xrange(1, mxitr + 1):
        XP = np.array(X)
        FP = F
        GP = np.array(G)
        dtXP = dtX[:]

        nls = 1
        deriv = rho * nrmG ** 2
        while True:
            if invH:
                X, infX = np.linlag.lstsq(np.eye(n) + tau * H, XP - tau * RX)[:2]
            else:
                aa, infR = np.linalg.lstsq(eye2k + (0.5 * tau) * VU, VX)[:2]
                X = XP - np.dot(U, tau * aa)
            F, G = fun(X, *args)
            nfe += 1
            if F <= Cval - tau * deriv or nls >= 5:
                break
            tau = eta * tau
            nls += 1

        GX = np.dot(G.T, X)
        if invH:
            GXT = np.dot(G, X.T)
            H = 0.5 * (GXT - GXT.T)
            RX = np.dot(H, X)
        else:
            if projG == 1:
                U = np.hstack([G, X])
                V = np.hstack([X, -G])
                VU = np.dot(V.T, U)
            elif projG == 2:
                GB = G - 0.5 * np.dot(X, np.dot(X.T, G))
                U = np.hstack([GB, X])
                V = np.hstack([X, -GB])
                VU = np.dot(V.T, U)
            VX = np.dot(V.T, X)
        dtX = G - np.dot(X, GX)
        nrmG = np.linalg.norm(dtX, 'fro')

        S = X - XP
        XDiff = np.linalg.norm(S, 'fro') / np.sqrt(n)

        tau = tau_max
        FDiff = np.abs(FP - F) / (np.abs(FP) + 1.)

        Y = dtX - dtXP
        SY = np.abs(np.sum(np.sum(S * Y)))
        if itr % 2 == 0:
            tau = np.sum(np.sum(S * S)) / SY
        else:
            tau = SY / np.sum(np.sum(Y * Y))
        
        tau = max(min(tau, 1e20), 1e-20)
  
        if disp:
            print '%4d \t %3.2e \t %4.3e \t %3.2e \t %3.2e \t %3.2e %2d' % (itr, tau, F, nrmG, XDiff, FDiff, nls)
        if (XDiff < xtol and FDiff < ftol) or nrmG < gtol:
            if (not nrmG == 0.) and itr <= 2:
                ftol = 0.1 * ftol
                xtol = 0.1 * xtol
                gtol = 0.1 * gtol
            elif FP < F:
                X = XP
                F = FP
                break
            else:
                success = True
                break
    Qp = Q
    Q = gamma * Qp + 1.
    Cval = (gamma * Qp * Cval + F) / Q
    class R(object):
        pass
    res = R()
    res.fun = F
    res.X = X
    res.nit = itr
    res.nfev = nfe
    res.success = success
    return res
