# -*- coding: utf-8 -*-
"""
Step 4 - look at the trained synapses as hardware.

Training produces a set of polynomial coefficients.  For the project those
coefficients are not just numbers: each one describes a characteristic that a
fabricated device has to realise.  This script asks the three questions that
decide whether the resistor-based approach is workable:

  1. What do the learned synapses look like?
     The polynomials are printed in readable a0 + a1*u + a2*u^2 + ... form and
     plotted, so they can be compared with the measured device curves from
     device_fit.py.

  2. How precisely do the coefficients have to be set?
     Every coefficient is quantised to a limited number of levels, which is
     what a resistor network with a finite set of available values gives you.
     The accuracy is then re-measured.

  3. How much component tolerance can the design absorb?
     Gaussian noise of a given relative size is added to every coefficient,
     many times over, and the spread of the resulting accuracy is reported.
     This is the number that matters for the resistor-versus-memristor
     argument: memristors drift, so their effective tolerance grows over time,
     while a resistor keeps whatever tolerance it was made with.

Run directly (F5 in Spyder).  Trains N_MODELS crossbars and then analyses them.
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader
from poly_basis import format_polynomial
from pnn import PNN

# --- settings --------------------------------------------------------------
SPLIT = config.SPLIT
DEGREE = config.POLY_DEGREE
QUANT_LEVELS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64]
TOLERANCES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
N_TRIALS = 40            # random draws per tolerance level
N_MODELS = 5             # independently trained crossbars, averaged over
#
# Averaging over several trained crossbars matters here.  A crossbar has only
# a handful of coefficients (17 for a 4x1 degree-3 layer), so rounding one
# particular set onto a grid can happen to help or hurt; a single run produces
# a visibly jagged curve.  Averaging over independently trained networks gives
# the trend that actually describes the design.
# ---------------------------------------------------------------------------


def train_crossbar(ds, degree=DEGREE, seed=0):
    net = PNN(n_in=ds.X_train.shape[1], n_out=1, hidden=(), degree=degree,
              basis=config.POLY_BASIS, task="regression", seed=seed)
    net.fit(ds.X_train, ds.y_reg_train, ds.X_val, ds.y_reg_val,
            epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE, l2=config.L2, clip=config.GRAD_CLIP,
            patience=config.PATIENCE, seed=seed, verbose=0)
    return net


def test_mae(net, ds):
    prediction = ds.target_scaler.inverse_transform(net.predict(ds.X_test).ravel())
    return float(np.mean(np.abs(prediction - ds.T_test)))


def quantise(coeffs, n_levels):
    """
    Round the coefficients onto `n_levels` equally spaced values.

    Each polynomial order gets its own grid, spanning the range that order
    actually needs.  That matches the hardware: the a0, a1, a2 ... terms of a
    device are set by different components, each with its own value range, so
    forcing all orders onto one shared grid would understate what a real
    resistor network can do.  The orders here differ by more than two decades
    (a1 ~ 1.4 against a2 ~ 0.005), so the distinction matters.
    """
    out = np.empty_like(coeffs)
    for p in range(coeffs.shape[-1]):
        values = coeffs[..., p]
        lo, hi = values.min(), values.max()
        if hi <= lo:
            out[..., p] = values
            continue
        step = (hi - lo) / (n_levels - 1)
        out[..., p] = lo + np.round((values - lo) / step) * step
    return out


# ---------------------------------------------------------------------------
# 1. what the synapses look like
# ---------------------------------------------------------------------------

def report_synapses(net):
    layer = net.poly_layers()[0]
    power = layer.power_coefficients()

    print("\nLearned synapses, written as characteristics P(u) with u the")
    print("normalised device current in [-1, 1]:\n")
    for i in range(layer.n_in):
        for j in range(layer.n_out):
            print("  %-12s -> out %d :  P(u) = %s"
                  % (config.DEVICE_LABELS[i], j,
                     format_polynomial(power[i, j], "u", 4)))
    return power


def plot_synapses(net):
    u, curves = net.synapse_curves(0)
    layer = net.poly_layers()[0]

    fig, axes = plt.subplots(1, layer.n_in, figsize=(3.4 * layer.n_in, 3.8),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        for j in range(layer.n_out):
            ax.plot(u, curves[i, j], linewidth=2,
                    label="out %d" % j if layer.n_out > 1 else None)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(config.DEVICE_LABELS[i], fontsize=11, fontweight="bold")
        ax.set_xlabel("normalised current u")
        ax.grid(True, alpha=0.3)
        if layer.n_out > 1 and layer.n_out <= 8:
            ax.legend(fontsize=7)
    axes[0].set_ylabel("synapse output P(u)")
    fig.suptitle("Characteristic each device has to realise (degree %d)"
                 % layer.degree, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.result_path("14_learned_synapse_curves.png"),
                dpi=config.DPI, bbox_inches="tight")
    return fig


def plot_coefficient_magnitudes(power):
    """How large is each order?  Tells you which terms the hardware must hold."""
    orders = power.shape[-1]
    magnitudes = np.abs(power).reshape(-1, orders)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    positions = np.arange(orders)
    ax.boxplot([magnitudes[:, p] for p in positions], positions=positions,
               widths=0.5)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(["a%d (u^%d)" % (p, p) for p in positions])
    ax.set_ylabel("|coefficient|")
    ax.set_title("Size of the polynomial coefficients by order",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(config.result_path("15_coefficient_magnitudes.png"),
                dpi=config.DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. quantisation
# ---------------------------------------------------------------------------

def quantisation_sweep(nets, ds):
    baselines = [test_mae(net, ds) for net in nets]

    rows = []
    for n_levels in QUANT_LEVELS:
        errors = []
        for net in nets:
            layer = net.poly_layers()[0]
            reference = layer.params["C"].copy()
            layer.params["C"] = quantise(reference, n_levels)
            errors.append(test_mae(net, ds))
            layer.params["C"] = reference
        errors = np.array(errors)
        rows.append({"levels": n_levels, "bits": np.log2(n_levels),
                     "MAE": errors.mean(), "std": errors.std()})
    return float(np.mean(baselines)), rows


# ---------------------------------------------------------------------------
# 3. component tolerance
# ---------------------------------------------------------------------------

def tolerance_sweep(nets, ds, seed=0):
    """
    Perturb every coefficient by Gaussian noise of the given relative size.

    The noise is scaled per polynomial order, for the same reason the
    quantisation grid is: a 1 % part is 1 % of whatever that order needs, not
    1 % of the largest coefficient in the layer.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for tolerance in TOLERANCES:
        errors = []
        for net in nets:
            layer = net.poly_layers()[0]
            reference = layer.params["C"].copy()
            scale = np.abs(reference).mean(axis=(0, 1), keepdims=True)
            for _ in range(N_TRIALS):
                layer.params["C"] = reference + rng.normal(
                    0.0, tolerance, size=reference.shape) * scale
                errors.append(test_mae(net, ds))
            layer.params["C"] = reference
        errors = np.array(errors)
        rows.append({"tolerance": tolerance, "mean": errors.mean(),
                     "std": errors.std(),
                     "p95": float(np.percentile(errors, 95))})
    return rows


def plot_hardware_limits(baseline, quant_rows, tol_rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    levels = [r["levels"] for r in quant_rows]
    axes[0].errorbar(levels, [r["MAE"] for r in quant_rows],
                     yerr=[r["std"] for r in quant_rows], fmt="o-",
                     color="#c0392b", linewidth=2, capsize=4,
                     label="mean over %d networks" % N_MODELS)
    axes[0].axhline(baseline, color="black", linestyle="--",
                    label="unquantised (%.2f °C)" % baseline)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(levels)
    axes[0].set_xticklabels([str(l) for l in levels])
    axes[0].set_xlabel("number of distinct coefficient values available")
    axes[0].set_ylabel("test MAE [°C]")
    axes[0].set_title("How finely the coefficients must be set",
                      fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    tolerances = [100 * r["tolerance"] for r in tol_rows]
    means = np.array([r["mean"] for r in tol_rows])
    stds = np.array([r["std"] for r in tol_rows])
    axes[1].plot(tolerances, means, "o-", color="#2980b9", linewidth=2,
                 label="mean over %d samples" % N_TRIALS)
    axes[1].fill_between(tolerances, means - stds, means + stds,
                         color="#2980b9", alpha=0.25, label="+/- 1 sd")
    axes[1].plot(tolerances, [r["p95"] for r in tol_rows], "s--",
                 color="#8e44ad", label="95th percentile")
    axes[1].axhline(baseline, color="black", linestyle="--",
                    label="nominal (%.2f °C)" % baseline)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("component tolerance [%]")
    axes[1].set_ylabel("test MAE [°C]")
    axes[1].set_title("How much component spread the design absorbs",
                      fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(config.result_path("16_hardware_limits.png"),
                dpi=config.DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------

def main():
    config.ensure_results_dir()
    ds = data_loader.prepare(split=SPLIT)

    print("=" * 78)
    print("THE TRAINED SYNAPSES AS HARDWARE")
    print("=" * 78)

    nets = [train_crossbar(ds, seed=seed) for seed in range(N_MODELS)]
    net = nets[0]
    baseline = float(np.mean([test_mae(n, ds) for n in nets]))
    print(net.summary())
    print("\nnominal test MAE: %.2f °C (mean of %d trained crossbars)"
          % (baseline, N_MODELS))

    power = report_synapses(net)

    print("\n" + "-" * 78)
    print("Coefficient quantisation")
    print("-" * 78)
    print("  %8s %8s %12s %9s %12s"
          % ("levels", "bits", "MAE[°C]", "sd", "vs nominal"))
    baseline_q, quant_rows = quantisation_sweep(nets, ds)
    for row in quant_rows:
        print("  %8d %8.1f %12.2f %9.2f %11.1f%%"
              % (row["levels"], row["bits"], row["MAE"], row["std"],
                 100.0 * (row["MAE"] - baseline_q) / baseline_q))

    usable = [r for r in quant_rows if r["MAE"] < 1.1 * baseline_q]
    if usable:
        cheapest = min(usable, key=lambda r: r["levels"])
        print("\n  %d distinct values per coefficient (%.1f bits) already stay "
              "within\n  10%% of the nominal error."
              % (cheapest["levels"], cheapest["bits"]))

    print("\n" + "-" * 78)
    print("Component tolerance (%d networks x %d random draws per level)"
          % (N_MODELS, N_TRIALS))
    print("-" * 78)
    print("  %10s %12s %10s %12s" % ("tolerance", "mean MAE", "sd", "95th pct"))
    tol_rows = tolerance_sweep(nets, ds)
    for row in tol_rows:
        print("  %9.1f%% %12.2f %10.2f %12.2f"
              % (100 * row["tolerance"], row["mean"], row["std"], row["p95"]))

    acceptable = [r for r in tol_rows if r["p95"] < 1.2 * baseline]
    if acceptable:
        worst = max(acceptable, key=lambda r: r["tolerance"])
        print("\n  up to %.1f%% component tolerance keeps 95%% of fabricated "
              "parts within\n  20%% of the nominal error."
              % (100 * worst["tolerance"]))

    plot_synapses(net)
    plot_coefficient_magnitudes(power)
    plot_hardware_limits(baseline, quant_rows, tol_rows)
    plt.show()

    print("\nFigures written to %s" % config.RESULTS_DIR)
    return nets, quant_rows, tol_rows


if __name__ == "__main__":
    models, quantisation, tolerance = main()
