# -*- coding: utf-8 -*-
"""
Polynomial bases used by the synaptic weight functions of the PNN.

Every synapse of the network is a *function* P_ij(u), not a scalar:

    P_ij(u) = sum_p  c_ijp * phi_p(u)

where phi_0 .. phi_P are the basis functions selected here.

Three bases are available:

    "power"      phi_p(u) = u^p                 (the natural, a + b*u + c*u^2 ... form)
    "chebyshev"  phi_p(u) = T_p(u)              (well conditioned on [-1, 1])
    "legendre"   phi_p(u) = L_p(u)              (orthogonal on [-1, 1])

Chebyshev/Legendre are recommended: the network inputs are scaled to [-1, 1]
and an orthogonal basis keeps the coefficients of different degrees from
fighting each other during training.  The learned coefficients can always be
converted back to the familiar power form with `to_power_basis` so the result
reads as  P(u) = a0 + a1*u + a2*u^2 + a3*u^3.
"""

import numpy as np
from numpy.polynomial import polynomial as _poly
from numpy.polynomial import chebyshev as _cheb
from numpy.polynomial import legendre as _leg

VALID_BASES = ("power", "chebyshev", "legendre")


def _basis_as_power_series(degree, basis):
    """
    Represent each basis function phi_p as power-series coefficients.

    Returns a list of 1-D arrays, lowest order first.
    """
    if basis not in VALID_BASES:
        raise ValueError("basis must be one of %s, got %r" % (VALID_BASES, basis))

    series = []
    for p in range(degree + 1):
        unit = np.zeros(p + 1)
        unit[p] = 1.0
        if basis == "power":
            coeffs = unit
        elif basis == "chebyshev":
            coeffs = _cheb.cheb2poly(unit)
        else:
            coeffs = _leg.leg2poly(unit)
        series.append(np.asarray(coeffs, dtype=float))
    return series


def basis_matrices(X, degree, basis):
    """
    Evaluate the basis and its derivative on every input value.

    Parameters
    ----------
    X : (n_samples, n_features) array, values expected in [-1, 1]
    degree : int
    basis : str

    Returns
    -------
    Phi  : (n_samples, n_features, degree+1)   Phi[n, i, p] = phi_p(X[n, i])
    dPhi : (n_samples, n_features, degree+1)   dPhi[n, i, p] = phi_p'(X[n, i])
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    series = _basis_as_power_series(degree, basis)

    Phi = np.empty((n_samples, n_features, degree + 1), dtype=float)
    dPhi = np.empty_like(Phi)

    for p, coeffs in enumerate(series):
        Phi[:, :, p] = _poly.polyval(X, coeffs)
        deriv = _poly.polyder(coeffs) if coeffs.size > 1 else np.zeros(1)
        dPhi[:, :, p] = _poly.polyval(X, deriv)

    return Phi, dPhi


def to_power_basis(coeffs, basis):
    """
    Convert coefficients expressed in `basis` to plain power-series form.

    `coeffs` may have any leading shape; the last axis is the degree axis.
    The returned array has the same shape, with the last axis holding
    [a0, a1, a2, ...] of  P(u) = a0 + a1*u + a2*u^2 + ...
    """
    coeffs = np.asarray(coeffs, dtype=float)
    degree = coeffs.shape[-1] - 1
    if basis == "power":
        return coeffs.copy()

    convert = _cheb.cheb2poly if basis == "chebyshev" else _leg.leg2poly
    flat = coeffs.reshape(-1, degree + 1)
    out = np.zeros_like(flat)
    for k in range(flat.shape[0]):
        converted = convert(flat[k])
        out[k, : converted.size] = converted
    return out.reshape(coeffs.shape)


def format_polynomial(power_coeffs, variable="u", decimals=3):
    """Pretty-print power-series coefficients as 'a0 + a1*u + a2*u^2 ...'."""
    terms = []
    for p, a in enumerate(np.atleast_1d(power_coeffs)):
        if abs(a) < 10.0 ** (-decimals):
            continue
        if p == 0:
            terms.append("%+.*f" % (decimals, a))
        elif p == 1:
            terms.append("%+.*f*%s" % (decimals, a, variable))
        else:
            terms.append("%+.*f*%s^%d" % (decimals, a, variable, p))
    return " ".join(terms) if terms else "0"


if __name__ == "__main__":
    # Quick self-check: T_2(u) = 2u^2 - 1, and its derivative is 4u.
    x = np.linspace(-1, 1, 5).reshape(-1, 1)
    Phi, dPhi = basis_matrices(x, 2, "chebyshev")
    assert np.allclose(Phi[:, 0, 2], 2 * x[:, 0] ** 2 - 1)
    assert np.allclose(dPhi[:, 0, 2], 4 * x[:, 0])
    print("chebyshev T_2 as power series:",
          format_polynomial(to_power_basis(np.array([0.0, 0.0, 1.0]), "chebyshev")))
    print("poly_basis self-check passed")
