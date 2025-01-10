from scipy.sparse.linalg._expm_multiply import LazyOperatorNormInfo,_fragment_3_1
import numpy as np
from scipy.linalg._decomp_qr import qr
from numba import njit

def traceest(A, m3, seed=None):
    rng = np.random.default_rng(seed)
    if len(A.shape) != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Expected A to be like a square matrix.")
    n = A.shape[-1]
    S = rng.choice([-1.0, +1.0], [n, m3])
    Q, _ = qr(A.matmat(S), overwrite_a=True, mode='economic')
    trQAQ = np.trace(Q.conj().T @ A.matmat(Q))
    G = rng.choice([-1, +1], [n, m3])
    right = G - Q@(Q.conj().T @ G)
    trGAG = np.trace(right.conj().T @ A.matmat(right))
    return trQAQ + trGAG/m3


def _exact_1_norm(A):
    # A compatibility function which should eventually disappear.

    return np.linalg.norm(A, 1)
def _ident_like(A):
    # A compatibility function which should eventually disappear.

    return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)
def _trace(A):
    # A compatibility function which should eventually disappear.
    return A.trace()

def expmv(A, B):
    X = _expm_multiply_simple(A, B, traceA=None)
    return X


def _expm_multiply_simple(A, B, t=1.0, traceA=None):
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    if traceA is None:
        # m3=1 is bit arbitrary choice, a more accurate trace (larger m3) might
        # speed up exponential calculation, but trace estimation is more costly
        traceA = _trace(A)
    mu = traceA / float(n)
    A = A - mu * ident
    A_1_norm = _exact_1_norm(A)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    return _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol)

@njit(cache=True)
def _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol=None):
    """
    A helper function.
    """
    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    F = B
    eta = np.exp(t*mu / float(s))
    for i in range(s):
        c1 = np.linalg.norm(B, np.inf)
        for j in range(m_star):
            coeff = t / float(s*(j+1))
            B = coeff * A.dot(B)
            c2 = np.linalg.norm(B, np.inf)
            F = F + B
            if c1 + c2 <= tol * np.linalg.norm(F,np.inf):
                break
            c1 = c2
        F = eta * F
        B = F
    return F
