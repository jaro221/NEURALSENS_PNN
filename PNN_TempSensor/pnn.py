# -*- coding: utf-8 -*-
"""
Physical Neural Network (PNN) with polynomial synaptic weights.

------------------------------------------------------------------------------
The model
------------------------------------------------------------------------------
In a conventional dense layer a synapse is one number:

    z_j = sum_i  w_ij * x_i + b_j

In the PNN a synapse is a *device*, and the device has a characteristic curve.
The measurements in this project show that the sensor current follows the
environment variable as a low-order polynomial, so the synapse is modelled as a
polynomial transfer function:

    z_j = sum_i  P_ij(x_i) + b_j        with   P_ij(u) = sum_p c_ijp * phi_p(u)

The trainable parameters are the polynomial coefficients c_ijp.  Physically
these describe the device characteristic that the resistor network has to
realise; they are fixed once the chip is made, which is exactly the
resistor-based (non-volatile) PNN the project proposes as a replacement for
memristor crossbars.

Two properties are worth noting:

  * degree 0 is the device leakage / offset term,
  * degree 1 is the ordinary linear weight.

So a PNN layer with degree=1 is *exactly* a standard dense layer.  That makes
the baseline comparison in the training scripts fair by construction: same
code, same initialisation, same optimiser, and the only thing that changes is
the polynomial degree.

------------------------------------------------------------------------------
Gradients
------------------------------------------------------------------------------
Everything is differentiated analytically and vectorised over the batch:

    dL/dc_ijp = sum_n  delta_nj * phi_p(x_ni)
    dL/dx_ni  = sum_j  delta_nj * sum_p c_ijp * phi_p'(x_ni)

The earlier prototype (../FCNN.py) approximated the polynomial-layer gradient
with np.polyfit of the error.  This module uses the exact gradient instead, and
it is orders of magnitude faster because no Python loop runs over samples.
Run this file directly to see the gradient check.
------------------------------------------------------------------------------
"""

import numpy as np

from poly_basis import basis_matrices, to_power_basis


# ===========================================================================
# Layers
# ===========================================================================

class PolyLayer(object):
    """
    Fully connected layer whose synapses are polynomials.

    Inputs are expected in [-1, 1]; use Tanh before stacking another PolyLayer.
    """

    def __init__(self, n_in, n_out, degree=3, basis="chebyshev",
                 rng=None, init_scale=None):
        self.n_in = n_in
        self.n_out = n_out
        self.degree = degree
        self.basis = basis
        rng = np.random.default_rng(0) if rng is None else rng

        # Scale like a standard He init and damp the higher orders, so training
        # starts close to a well-behaved linear network.
        if init_scale is None:
            init_scale = np.sqrt(2.0 / n_in)
        decay = 1.0 / (1.0 + np.arange(degree + 1))
        C = rng.normal(0.0, init_scale, size=(n_in, n_out, degree + 1))
        self.params = {"C": C * decay, "b": np.zeros((1, n_out))}
        self.grads = {}
        self._cache = None

    def forward(self, X):
        Phi, dPhi = basis_matrices(X, self.degree, self.basis)
        self._cache = (Phi, dPhi)
        return np.einsum("nip,iop->no", Phi, self.params["C"]) + self.params["b"]

    def backward(self, dZ):
        Phi, dPhi = self._cache
        self.grads["C"] = np.einsum("nip,no->iop", Phi, dZ)
        self.grads["b"] = dZ.sum(axis=0, keepdims=True)
        # dX[n,i] = sum_j dZ[n,j] * sum_p C[i,j,p] * phi_p'(x_ni)
        slope = np.einsum("nip,iop->nio", dPhi, self.params["C"])
        return np.einsum("nio,no->ni", slope, dZ)

    def power_coefficients(self):
        """Learned synapses as plain a0 + a1*u + a2*u^2 ... coefficients."""
        return to_power_basis(self.params["C"], self.basis)

    def describe(self):
        return "PolyLayer(%d -> %d, degree=%d, basis=%s)" % (
            self.n_in, self.n_out, self.degree, self.basis)


class DenseLayer(object):
    """Standard linear layer, used for the read-out."""

    def __init__(self, n_in, n_out, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        self.n_in = n_in
        self.n_out = n_out
        self.params = {
            "W": rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)),
            "b": np.zeros((1, n_out)),
        }
        self.grads = {}
        self._X = None

    def forward(self, X):
        self._X = X
        return X.dot(self.params["W"]) + self.params["b"]

    def backward(self, dZ):
        self.grads["W"] = self._X.T.dot(dZ)
        self.grads["b"] = dZ.sum(axis=0, keepdims=True)
        return dZ.dot(self.params["W"].T)

    def describe(self):
        return "DenseLayer(%d -> %d)" % (self.n_in, self.n_out)


class Tanh(object):
    """
    Saturating neuron.

    Besides being a good activation it maps the signal back into (-1, 1),
    which is the domain the polynomial synapses of the next layer expect.
    """

    def __init__(self):
        self.params = {}
        self.grads = {}
        self._A = None

    def forward(self, X):
        self._A = np.tanh(X)
        return self._A

    def backward(self, dZ):
        return dZ * (1.0 - self._A ** 2)

    def describe(self):
        return "Tanh()"


class ReLU(object):

    def __init__(self):
        self.params = {}
        self.grads = {}
        self._mask = None

    def forward(self, X):
        self._mask = X > 0
        return X * self._mask

    def backward(self, dZ):
        return dZ * self._mask

    def describe(self):
        return "ReLU()"


# ===========================================================================
# Optimiser
# ===========================================================================

class Adam(object):
    """Adam with bias correction, L2 and global-norm gradient clipping."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                 l2=0.0, clip=None):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.l2 = l2
        self.clip = clip
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1

        if self.clip is not None:
            total = 0.0
            for layer in layers:
                for g in layer.grads.values():
                    total += float(np.sum(g ** 2))
            norm = np.sqrt(total)
            if norm > self.clip:
                scale = self.clip / (norm + 1e-12)
                for layer in layers:
                    for name in layer.grads:
                        layer.grads[name] = layer.grads[name] * scale

        for li, layer in enumerate(layers):
            for name, grad in layer.grads.items():
                key = (li, name)
                param = layer.params[name]

                if self.l2 and name != "b":
                    grad = grad + self.l2 * param

                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ===========================================================================
# Losses
# ===========================================================================

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y_int):
    """Returns (loss, dlogits).  y_int holds integer class labels."""
    n = logits.shape[0]
    probs = softmax(logits)
    loss = -np.mean(np.log(np.clip(probs[np.arange(n), y_int], 1e-15, 1.0)))
    d = probs.copy()
    d[np.arange(n), y_int] -= 1.0
    return loss, d / n


def mse(pred, y):
    """Returns (loss, dpred).  y has shape (n, n_out)."""
    diff = pred - y
    return float(np.mean(diff ** 2)), 2.0 * diff / diff.size


# ===========================================================================
# Model
# ===========================================================================

class PNN(object):
    """
    Feed-forward network whose first (and optionally hidden) layers use
    polynomial synapses.

    task : "classification" -> softmax cross-entropy, metric = accuracy
           "regression"     -> mean squared error,    metric = MAE
    """

    def __init__(self, n_in, n_out, hidden=(16,), degree=3, basis="chebyshev",
                 task="classification", poly_hidden=False, activation="tanh",
                 seed=0):
        self.task = task
        self.n_in = n_in
        self.n_out = n_out
        self.degree = degree
        self.basis = basis
        self.best_epoch = None
        rng = np.random.default_rng(seed)

        act_cls = Tanh if activation == "tanh" else ReLU
        self.layers = []

        if not hidden:
            # One crossbar, no hidden neurons: the inputs drive the polynomial
            # devices and the column currents are the output.  This is the
            # layout the project actually proposes to fabricate, and it is the
            # configuration in which the device characteristic - rather than a
            # hidden activation function - has to provide all the nonlinearity.
            self.layers.append(PolyLayer(n_in, n_out, degree, basis, rng))
        else:
            prev = n_in
            for k, width in enumerate(hidden):
                if k == 0 or poly_hidden:
                    self.layers.append(PolyLayer(prev, width, degree, basis, rng))
                else:
                    self.layers.append(DenseLayer(prev, width, rng))
                self.layers.append(act_cls())
                prev = width
            self.layers.append(DenseLayer(prev, n_out, rng))

        self.history = {"train_loss": [], "val_loss": [],
                        "train_metric": [], "val_metric": []}

    # -- plumbing ----------------------------------------------------------

    def summary(self):
        lines = ["PNN (%s), degree=%d, basis=%s" % (self.task, self.degree, self.basis)]
        for layer in self.layers:
            lines.append("  " + layer.describe())
        lines.append("  trainable parameters: %d" % self.n_parameters())
        return "\n".join(lines)

    def n_parameters(self):
        return sum(p.size for layer in self.layers for p in layer.params.values())

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def _loss(self, logits, y):
        if self.task == "classification":
            return softmax_cross_entropy(logits, y)
        return mse(logits, y)

    def _metric(self, logits, y):
        if self.task == "classification":
            return float(np.mean(np.argmax(logits, axis=1) == y))
        return float(np.mean(np.abs(logits - y)))

    def get_params(self):
        return [{k: v.copy() for k, v in layer.params.items()} for layer in self.layers]

    def set_params(self, snapshot):
        for layer, saved in zip(self.layers, snapshot):
            for k, v in saved.items():
                layer.params[k] = v.copy()

    # -- training ----------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=300,
            batch_size=32, lr=0.01, l2=0.0, clip=5.0, patience=None,
            seed=0, verbose=25):
        """
        Train with mini-batch Adam.

        If a validation set is given, the parameters of the best validation
        epoch are restored at the end, and early stopping applies when
        `patience` is set.
        """
        opt = Adam(lr=lr, l2=l2, clip=clip)
        rng = np.random.default_rng(seed)
        n = X_train.shape[0]

        best_val = np.inf
        best_snapshot = self.get_params()
        best_epoch = 0
        since_improved = 0
        val_metric = np.nan

        for epoch in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                logits = self.forward(X_train[idx])
                _, dlogits = self._loss(logits, y_train[idx])
                self.backward(dlogits)
                opt.step(self.layers)

            train_logits = self.forward(X_train)
            train_loss, _ = self._loss(train_logits, y_train)
            self.history["train_loss"].append(train_loss)
            self.history["train_metric"].append(self._metric(train_logits, y_train))

            if X_val is not None:
                val_logits = self.forward(X_val)
                val_loss, _ = self._loss(val_logits, y_val)
                val_metric = self._metric(val_logits, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_metric"].append(val_metric)

                if val_loss < best_val - 1e-9:
                    best_val = val_loss
                    best_snapshot = self.get_params()
                    best_epoch = epoch
                    since_improved = 0
                else:
                    since_improved += 1
            else:
                val_loss = np.nan

            if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
                msg = "  epoch %4d  train_loss=%.5f" % (epoch, train_loss)
                if X_val is not None:
                    msg += "  val_loss=%.5f  val_metric=%.4f" % (val_loss, val_metric)
                print(msg)

            if patience is not None and since_improved >= patience:
                if verbose:
                    print("  early stop at epoch %d (best epoch %d)" % (epoch, best_epoch))
                break

        if X_val is not None:
            self.set_params(best_snapshot)
            self.best_epoch = best_epoch
        return self.history

    # -- inference ---------------------------------------------------------

    def predict_logits(self, X):
        return self.forward(X)

    def predict(self, X):
        out = self.forward(X)
        if self.task == "classification":
            return np.argmax(out, axis=1)
        return out

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is only defined for classification")
        return softmax(self.forward(X))

    # -- interpretation ----------------------------------------------------

    def poly_layers(self):
        return [l for l in self.layers if isinstance(l, PolyLayer)]

    def synapse_curves(self, layer_index=0, n_points=200):
        """
        Evaluate every synapse of a polynomial layer on a grid over [-1, 1].

        Returns (u, curves) with curves of shape (n_in, n_out, n_points):
        the characteristic each physical device has to realise.
        """
        layer = self.poly_layers()[layer_index]
        u = np.linspace(-1.0, 1.0, n_points)
        Phi, _ = basis_matrices(u.reshape(-1, 1), layer.degree, layer.basis)
        curves = np.einsum("np,iop->ion", Phi[:, 0, :], layer.params["C"])
        return u, curves


# ===========================================================================
# Gradient check
# ===========================================================================

def gradient_check(seed=3):
    """Compare analytic gradients against central finite differences."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(7, 4))
    y = rng.integers(0, 3, size=7)

    net = PNN(4, 3, hidden=(5,), degree=3, task="classification",
              poly_hidden=True, seed=seed)

    logits = net.forward(X)
    _, dlogits = net._loss(logits, y)
    net.backward(dlogits)
    analytic_all = [{k: v.copy() for k, v in layer.grads.items()}
                    for layer in net.layers]

    worst = 0.0
    for layer, analytic in zip(net.layers, analytic_all):
        for name, param in layer.params.items():
            flat = param.ravel()
            for _ in range(12):
                k = int(rng.integers(0, flat.size))
                original = flat[k]
                eps = 1e-6

                flat[k] = original + eps
                loss_plus, _ = net._loss(net.forward(X), y)
                flat[k] = original - eps
                loss_minus, _ = net._loss(net.forward(X), y)
                flat[k] = original

                numeric = (loss_plus - loss_minus) / (2 * eps)
                a = analytic[name].ravel()[k]
                rel = abs(numeric - a) / max(1e-8, abs(numeric) + abs(a))
                worst = max(worst, rel)
    return worst


if __name__ == "__main__":
    print("Gradient check on a degree-3 PNN with a polynomial hidden layer...")
    worst_error = gradient_check()
    print("worst relative error (analytic vs numeric): %.3e" % worst_error)
    assert worst_error < 1e-5, "gradient check FAILED"
    print("gradient check passed")

    # A degree-1 PolyLayer must reproduce a dense layer exactly.
    _rng = np.random.default_rng(0)
    _X = _rng.uniform(-1, 1, size=(5, 3))
    _layer = PolyLayer(3, 2, degree=1, basis="power", rng=_rng)
    _W = _layer.params["C"][:, :, 1]
    _offset = _layer.params["C"][:, :, 0].sum(axis=0)
    assert np.allclose(_layer.forward(_X), _X.dot(_W) + _offset)
    print("degree-1 PolyLayer == DenseLayer: confirmed")
