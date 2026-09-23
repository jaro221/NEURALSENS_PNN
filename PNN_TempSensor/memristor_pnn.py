# -*- coding: utf-8 -*-
"""
The PNN with weights that ARE the memristors.

------------------------------------------------------------------------------
Why this file exists
------------------------------------------------------------------------------
In pnn.py the synapse is a polynomial of the *signal*, P(u), with freely
learned coefficients.  That is a useful function approximator but it is not
tied to the device: nothing in it refers to the memristor at all.

Here the synapse is the device.  Its conductance follows the compact model of
the manuscript,

    I(V,T) = u(V) * V * C_pf(V) * exp( -Ea(V) / (kB*T) )                   (1)

so the weight a crossbar element applies is

    g(T) = I(V,T)/V = C_pf * exp( -Ea / (kB*T) ).                          (2)

The trainable parameters are C_pf and Ea, i.e. the *programmed state* of the
device, which is what a SET pulse actually changes.  Training therefore answers
a hardware question - which conductance state must each device be written to -
rather than producing abstract numbers.

------------------------------------------------------------------------------
The polynomial weight
------------------------------------------------------------------------------
Equation (2) is an exponential in 1/T and cannot be evaluated by a polynomial
multiply-accumulate.  Over the measured window it does not need to be: with
tau the temperature normalised to [-1,1] over 30..150 °C,

    g(T) ~= P(tau) = sum_p c_p * phi_p(tau)

is accurate to better than 0.05 % at the activation energies measured here
(Ea = 0.195..0.224 eV).  `approximation_report()` prints the table.

So the polynomial weight of pnn.py is recovered, but now it is *derived* from
(2) rather than postulated, and its coefficients are a function of the
programmed state.  Degree 3 is not a tuning choice: it is the order at which
the approximation error of (2) drops below the measurement noise.

------------------------------------------------------------------------------
Signed weights
------------------------------------------------------------------------------
A conductance is positive, so a single device cannot realise a negative weight
and a purely positive crossbar can only produce responses that are monotonic in
temperature.  Each weight is therefore a differential pair, as is standard in
memristive crossbars,

    w_ij(T) = g+_ij(T) - g-_ij(T),

read on two columns whose currents are subtracted.  A pair of devices with
different Ea gives a non-monotonic response, which is what lets a column select
a temperature band.

Run directly (F5 in Spyder).
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader

KB = 8.617333262e-5          # eV/K
T_MIN_C, T_MAX_C = 30.0, 150.0


# ===========================================================================
# temperature normalisation
# ===========================================================================

def to_tau(temp_c):
    """Map °C onto [-1, 1], the domain of the polynomial surrogate."""
    return 2.0 * (np.asarray(temp_c, float) - T_MIN_C) / (T_MAX_C - T_MIN_C) - 1.0


def to_kelvin(temp_c):
    return np.asarray(temp_c, float) + 273.15


# ===========================================================================
# 1. the device model of equation (1) / (2)
# ===========================================================================

def conductance(C_pf, Ea, temp_c):
    """g(T) = C_pf * exp(-Ea / kB T), equation (2)."""
    return C_pf * np.exp(-Ea / (KB * to_kelvin(temp_c)))


def fit_device_states(data=None, report=True):
    """
    Extract (C_pf, Ea) for each measured channel from the sweep.

    These are the real programmed states of the devices that were measured, and
    they set the parameter range a trainable crossbar may use.
    """
    data = data_loader.load_raw() if data is None else data
    temp_c = data[config.TEMP_COLUMN].values
    inv_kt = 1.0 / (KB * to_kelvin(temp_c))

    states = {}
    if report:
        print("Programmed states extracted from the measurement, via (2):")
        print("  %-12s %8s %12s %10s" % ("channel", "Ea[eV]", "C_pf[S]", "R2"))
    for column, label, volt in zip(config.CURRENT_COLUMNS,
                                   config.DEVICE_LABELS,
                                   config.BIAS_VOLTAGES):
        current = data[column].values
        keep = current > 1e-9
        g = current[keep] / volt
        slope, intercept = np.polyfit(inv_kt[keep], np.log(g), 1)
        Ea, C_pf = -slope, np.exp(intercept)

        predicted = np.polyval([slope, intercept], inv_kt[keep])
        residual = np.log(g) - predicted
        r2 = 1.0 - np.sum(residual ** 2) / np.sum((np.log(g) - np.log(g).mean()) ** 2)

        states[column] = {"Ea": Ea, "C_pf": C_pf, "V": volt, "R2": r2,
                          "label": label}
        if report:
            print("  %-12s %8.4f %12.3e %10.4f" % (label, Ea, C_pf, r2))
    return states


# ===========================================================================
# 2. the polynomial surrogate of the device weight
# ===========================================================================

def measured_envelope(states=None, widen=1.0):
    """
    The programmable envelope the measurement actually demonstrates.

    A crossbar may only be given states the fabricated devices can reach, so
    the bounds on (C_pf, Ea) are taken from the sweep rather than assumed.
    Leaving them open is not a harmless convenience: unbounded, training drives
    C_pf to order 1e5 S - five decades past anything fabricable - and reports
    an accuracy no physical array could reach.

    `widen` > 1 expands the window geometrically about its centre, to ask what
    a broader programmable range would buy.  On this data it buys very little
    (see the docstring of ArrheniusCrossbar).

    Returns (ea_range, c_range).

    Caveat worth keeping in view: the four channels are four read *voltages*,
    so strictly this is the envelope of C_pf(V) and Ea(V) for the state the
    devices happened to be in, not a survey of programmable states.  The
    multi-state measurement is what would establish the latter.
    """
    states = fit_device_states(report=False) if states is None else states
    Ea = np.array([s["Ea"] for s in states.values()])
    C = np.array([s["C_pf"] for s in states.values()])

    def expand(lo, hi, log=False):
        if log:
            centre = np.sqrt(lo * hi)
            return centre * (lo / centre) ** widen, centre * (hi / centre) ** widen
        centre = 0.5 * (lo + hi)
        return centre + (lo - centre) * widen, centre + (hi - centre) * widen

    return expand(Ea.min(), Ea.max()), expand(C.min(), C.max(), log=True)


def polynomial_weight(C_pf, Ea, degree=3, n_points=400):
    """
    Least-squares polynomial surrogate of g(T) on tau in [-1, 1].

    Returns coefficients in numpy's highest-order-first convention, which is
    what np.polyval expects.
    """
    temp_c = np.linspace(T_MIN_C, T_MAX_C, n_points)
    return np.polyfit(to_tau(temp_c), conductance(C_pf, Ea, temp_c), degree)


def approximation_error(C_pf, Ea, degree=3, n_points=400):
    """Maximum relative error of the polynomial surrogate, in percent."""
    temp_c = np.linspace(T_MIN_C, T_MAX_C, n_points)
    exact = conductance(C_pf, Ea, temp_c)
    approx = np.polyval(polynomial_weight(C_pf, Ea, degree, n_points),
                        to_tau(temp_c))
    return 100.0 * np.max(np.abs(approx - exact)) / np.max(np.abs(exact))


def approximation_report(degrees=(1, 2, 3, 4), energies=None):
    """How well a polynomial can stand in for equation (2)."""
    if energies is None:
        energies = [0.15, 0.20, 0.225, 0.30, 0.40, 0.60]
    print("\nMaximum relative error of the polynomial surrogate of (2),")
    print("over %.0f-%.0f °C:\n" % (T_MIN_C, T_MAX_C))
    print("  %8s" % "Ea[eV]" + "".join("%12s" % ("degree %d" % d) for d in degrees))
    for Ea in energies:
        row = "".join("%11.4f%%" % approximation_error(1.0, Ea, d) for d in degrees)
        print("  %8.3f" % Ea + row)
    print("\n  measured devices sit at Ea = 0.195-0.224 eV, where degree 3")
    print("  keeps the error near 0.03 % - far below the read-out noise.")


# ===========================================================================
# 3. a crossbar whose weights are devices
# ===========================================================================

class ArrheniusCrossbar(object):
    """
    Crossbar of memristive weights, each a differential pair of devices.

    Every element carries two devices with their own programmed state:

        w_ij(T) = C+_ij exp(-E+_ij / kB T)  -  C-_ij exp(-E-_ij / kB T)

    and the column current for an applied read pattern x is

        z_j(T) = sum_i x_i * w_ij(T).

    The trainable parameters are the programmed states themselves.  C is
    parameterised as exp(c) so it stays positive, and both parameters are
    clipped to the envelope the fabricated devices are known to reach, which
    `measured_envelope()` derives from the sweep.  Widening that envelope
    threefold or tenfold was tested and changes nothing outside the spread
    over initialisations at 6 bands, and gains at most two points at 4 bands:
    the states already demonstrated are sufficient for this task.

    `mode` selects how the weight is evaluated:
        "exact"       equation (2) directly
        "polynomial"  the degree-D surrogate, i.e. what a polynomial
                      multiply-accumulate would compute
    """

    def __init__(self, n_in, n_out, ea_range=None, c_range=None,
                 degree=3, seed=0, scale=1.0):
        if ea_range is None or c_range is None:
            measured_ea, measured_c = measured_envelope()
            ea_range = measured_ea if ea_range is None else ea_range
            c_range = measured_c if c_range is None else c_range
        rng = np.random.default_rng(seed)
        self.n_in, self.n_out = n_in, n_out
        self.ea_lo, self.ea_hi = ea_range
        # C_pf must stay inside what a device can actually be programmed to.
        # Without this bound the optimiser drives C_pf to ~1e5 S, five decades
        # beyond anything fabricable, and the accuracy it then reports is not
        # achievable in hardware.
        self.c_lo, self.c_hi = np.log(c_range[0]), np.log(c_range[1])
        self.degree = degree
        # Read-out gain.  A crossbar column is read through a transimpedance
        # amplifier, so the absolute current scale is set by the circuit, not
        # by the devices.  Keeping it as an explicit factor means the bounded
        # conductances are not asked to supply the logit scale as well.
        self.scale = scale

        shape = (n_in, n_out)
        mid = 0.5 * (self.ea_lo + self.ea_hi)
        self.Ea_p = rng.uniform(self.ea_lo, self.ea_hi, shape)
        self.Ea_n = rng.uniform(self.ea_lo, self.ea_hi, shape)
        # start both arms at a comparable conductance at mid-range
        self.c_p = np.log(rng.uniform(0.5, 1.5, shape)) + mid / (KB * 360.0)
        self.c_n = np.log(rng.uniform(0.5, 1.5, shape)) + mid / (KB * 360.0)
        self.bias = np.zeros((1, n_out))

        self.params = ["c_p", "Ea_p", "c_n", "Ea_n", "bias"]
        self.grads = {}

    # -- weights ---------------------------------------------------------

    def _arm(self, c, Ea, temp_c, mode):
        if mode == "exact":
            return np.exp(c)[None, :, :] * np.exp(
                -Ea[None, :, :] / (KB * to_kelvin(temp_c)[:, None, None]))
        # polynomial surrogate, fitted per element on the fly
        tau = to_tau(temp_c)
        out = np.empty((len(tau), self.n_in, self.n_out))
        for i in range(self.n_in):
            for j in range(self.n_out):
                coeffs = polynomial_weight(np.exp(c[i, j]), Ea[i, j], self.degree)
                out[:, i, j] = np.polyval(coeffs, tau)
        return out

    def weights(self, temp_c, mode="exact"):
        """w_ij(T) for every sample, shape (n_samples, n_in, n_out)."""
        return (self._arm(self.c_p, self.Ea_p, temp_c, mode)
                - self._arm(self.c_n, self.Ea_n, temp_c, mode))

    # -- forward / backward ----------------------------------------------

    def forward(self, x, temp_c, mode="exact"):
        """
        x : (n_in,) applied read pattern, or (n_samples, n_in)
        returns column currents, shape (n_samples, n_out)
        """
        x = np.atleast_2d(x)
        if x.shape[0] == 1:
            x = np.repeat(x, len(np.atleast_1d(temp_c)), axis=0)

        self._x, self._temp_c = x, temp_c
        self._gp = self._arm(self.c_p, self.Ea_p, temp_c, mode)
        self._gn = self._arm(self.c_n, self.Ea_n, temp_c, mode)
        w = self._gp - self._gn
        return self.scale * np.einsum("ni,nio->no", x, w) + self.bias

    def backward(self, dz):
        """Gradients with respect to the programmed states."""
        dz = self.scale * dz
        inv_kt = 1.0 / (KB * to_kelvin(self._temp_c))

        # dL/dw_ij = sum_n x_ni * dz_nj
        common = np.einsum("ni,no->nio", self._x, dz)

        self.grads["c_p"] = np.einsum("nio,nio->io", common, self._gp)
        self.grads["c_n"] = -np.einsum("nio,nio->io", common, self._gn)
        self.grads["Ea_p"] = -np.einsum("nio,nio,n->io", common, self._gp, inv_kt)
        self.grads["Ea_n"] = np.einsum("nio,nio,n->io", common, self._gn, inv_kt)
        self.grads["bias"] = dz.sum(axis=0, keepdims=True)
        return self.grads

    def clip_to_physical(self):
        """Keep the programmed states inside what the devices can reach."""
        np.clip(self.Ea_p, self.ea_lo, self.ea_hi, out=self.Ea_p)
        np.clip(self.Ea_n, self.ea_lo, self.ea_hi, out=self.Ea_n)
        np.clip(self.c_p, self.c_lo, self.c_hi, out=self.c_p)
        np.clip(self.c_n, self.c_lo, self.c_hi, out=self.c_n)

    def calibrate_gain(self, x, temp_c, target=3.0):
        """
        Set the read-out gain so the column currents span a usable range.

        This is the transimpedance amplifier, not a weight: one global number
        for the whole array.  Without it the bounded conductances produce
        column currents of order 1e-3, the softmax is flat, and no gradient
        reaches the devices at all.
        """
        z = self.forward(x, temp_c, mode="exact")
        self.scale *= target / max(float(np.std(z)), 1e-30)
        return self.scale

    def n_devices(self):
        return 2 * self.n_in * self.n_out


# ===========================================================================
# 4. training
# ===========================================================================

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(z, y):
    n = z.shape[0]
    p = softmax(z)
    loss = -np.mean(np.log(np.clip(p[np.arange(n), y], 1e-15, 1.0)))
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


def train_crossbar(crossbar, x_read, temp_train, y_train, temp_val, y_val,
                   epochs=400, lr=0.02, verbose=50, recalibrate_every=250):
    """
    Adam on the programmed states.

    The read-out gain is re-calibrated periodically.  The devices are bounded,
    so they cannot grow to supply the logit scale themselves; the gain is what
    a transimpedance amplifier would provide and it is one number for the whole
    array, not a per-device parameter.
    """
    m = {k: np.zeros_like(getattr(crossbar, k)) for k in crossbar.params}
    v = {k: np.zeros_like(getattr(crossbar, k)) for k in crossbar.params}
    b1, b2, eps = 0.9, 0.999, 1e-8

    best = (np.inf, None)
    step_count = 0
    history = []
    for epoch in range(1, epochs + 1):
        if recalibrate_every and (epoch - 1) % recalibrate_every == 0:
            crossbar.calibrate_gain(x_read, temp_train)
            # Rescaling the gain rescales the loss surface, so the Adam moment
            # estimates and the best-so-far validation loss are no longer
            # comparable.  Both are reset; without this the stale momentum
            # fights the new scale and training stalls near chance level.
            for k in crossbar.params:
                m[k][...] = 0.0
                v[k][...] = 0.0
            step_count = 0
            best = (np.inf, {k: getattr(crossbar, k).copy()
                             for k in crossbar.params})

        step_count += 1
        z = crossbar.forward(x_read, temp_train, mode="exact")
        loss, dz = cross_entropy(z, y_train)
        crossbar.backward(dz)

        for k in crossbar.params:
            g = crossbar.grads[k]
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g ** 2
            step = lr * (m[k] / (1 - b1 ** step_count)) / (
                np.sqrt(v[k] / (1 - b2 ** step_count)) + eps)
            setattr(crossbar, k, getattr(crossbar, k) - step)
        crossbar.clip_to_physical()

        zv = crossbar.forward(x_read, temp_val, mode="exact")
        val_loss, _ = cross_entropy(zv, y_val)
        val_acc = float(np.mean(np.argmax(zv, axis=1) == y_val))
        history.append((loss, val_loss, val_acc))
        if val_loss < best[0]:
            best = (val_loss, {k: getattr(crossbar, k).copy()
                               for k in crossbar.params})
        if verbose and epoch % verbose == 0:
            print("  epoch %4d  loss %.4f  val_loss %.4f  val_acc %.4f"
                  % (epoch, loss, val_loss, val_acc))

    for k, value in best[1].items():
        setattr(crossbar, k, value)
    return history


# ===========================================================================

def main():
    config.ensure_results_dir()
    print("=" * 78)
    print("PNN WITH MEMRISTIVE WEIGHTS  -  weights follow equation (1)")
    print("=" * 78)

    states = fit_device_states()
    approximation_report()

    # ---- the measured devices, exact vs polynomial --------------------
    print("\nPer measured device, degree-3 surrogate of its own state:")
    for column, st in states.items():
        print("  %-12s Ea=%.3f eV   max error %.4f %%"
              % (st["label"], st["Ea"],
                 approximation_error(st["C_pf"], st["Ea"], 3)))

    # ---- train a crossbar ---------------------------------------------
    ds = data_loader.prepare(split="ramp", n_classes=config.N_CLASSES)
    x_read = np.asarray(config.BIAS_VOLTAGES, float)      # the applied pattern

    ea_range, c_range = measured_envelope()
    print("\nProgrammable envelope taken from the measurement:")
    print("  Ea   %.4f to %.4f eV" % ea_range)
    print("  C_pf %.2e to %.2e S  (ratio %.0fx)"
          % (c_range[0], c_range[1], c_range[1] / c_range[0]))
    crossbar = ArrheniusCrossbar(n_in=4, n_out=ds.n_classes,
                                 ea_range=ea_range, c_range=c_range,
                                 degree=3, seed=0)
    print("\nCrossbar: %d x %d elements, %d devices (differential pairs),"
          % (4, ds.n_classes, crossbar.n_devices()))
    print("          %d trainable numbers (C_pf and Ea per device)."
          % (2 * crossbar.n_devices() + ds.n_classes))

    print("\nTraining the programmed states...")
    train_crossbar(crossbar, x_read, ds.T_train, ds.y_class_train,
                   ds.T_val, ds.y_class_val, epochs=1500, lr=0.02, verbose=300)
    print("  read-out gain after training: %.3e" % crossbar.scale)
    print("  programmed C_pf spans %.2e to %.2e S"
          % (min(np.exp(crossbar.c_p).min(), np.exp(crossbar.c_n).min()),
             max(np.exp(crossbar.c_p).max(), np.exp(crossbar.c_n).max())))

    # ---- evaluate, exact weights vs polynomial weights ----------------
    print("\n" + "-" * 78)
    print("Test set = the cooling curve (%d samples)" % len(ds.T_test))
    print("-" * 78)
    accuracies = {}
    for mode in ("exact", "polynomial"):
        z = crossbar.forward(x_read, ds.T_test, mode=mode)
        pred = np.argmax(z, axis=1)
        acc = float(np.mean(pred == ds.y_class_test))
        near = float(np.mean(np.abs(pred - ds.y_class_test) <= 1))
        accuracies[mode] = acc
        print("  %-11s weights:  accuracy %.4f   within 1 band %.4f"
              % (mode, acc, near))
    print("\n  replacing equation (2) by its degree-3 polynomial changes")
    print("  the accuracy by %.4f - the surrogate is free."
          % abs(accuracies["exact"] - accuracies["polynomial"]))

    make_figure(crossbar, states, ds)
    plt.show()
    return crossbar, states, ds


def make_figure(crossbar, states, ds):
    temp_c = np.linspace(T_MIN_C, T_MAX_C, 300)
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) measured device conductance and its polynomial surrogate
    ax = axes[0]
    for (column, st), colour in zip(states.items(), ramp):
        g = conductance(st["C_pf"], st["Ea"], temp_c)
        ax.semilogy(temp_c, g, color=colour, linewidth=2,
                    label="%s, $E_a$=%.3f eV" % (st["label"], st["Ea"]))
        surrogate = np.polyval(polynomial_weight(st["C_pf"], st["Ea"], 3),
                               to_tau(temp_c))
        ax.semilogy(temp_c, surrogate, color="black", linewidth=0.9,
                    linestyle=(0, (4, 3)))
    ax.set_xlabel("temperature [°C]")
    ax.set_ylabel("conductance $g(T)$ [S]")
    ax.set_title("(a) equation (2) and its degree-3\npolynomial surrogate (dashed)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # (b) approximation error against degree
    ax = axes[1]
    for degree, colour in zip((1, 2, 3, 4), ["#c0392b", "#e67e22", "#27ae60",
                                             "#2980b9"]):
        energies = np.linspace(0.10, 0.45, 40)
        err = [approximation_error(1.0, Ea, degree) for Ea in energies]
        ax.semilogy(energies, err, color=colour, linewidth=2,
                    label="degree %d" % degree)
    ax.axvspan(0.195, 0.224, color="#95a5a6", alpha=0.25, linewidth=0)
    ax.text(0.2095, 40, "measured\ndevices", ha="center", fontsize=7.5)
    ax.set_xlabel("activation energy $E_a$ [eV]")
    ax.set_ylabel("max. relative error [%]")
    ax.set_title("(b) how well a polynomial can\nstand in for equation (2)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (c) the trained weights of the crossbar
    ax = axes[2]
    w = crossbar.weights(temp_c, mode="exact")
    for j in range(min(crossbar.n_out, 12)):
        ax.plot(temp_c, w[:, 0, j], linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("temperature [°C]")
    ax.set_ylabel("weight $w_{1j}(T)$ [S]")
    ax.set_title("(c) trained differential weights\nof input 1 (one line per band)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = config.result_path("20_memristive_weights.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("\nFigure written: %s" % path)


if __name__ == "__main__":
    crossbar, states, dataset = main()
