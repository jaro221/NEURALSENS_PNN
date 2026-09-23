# -*- coding: utf-8 -*-
"""
Step 1 - characterise the four measured devices as polynomials.

This is the experimental justification for using polynomial synapses at all.
Two families of polynomials are extracted from the sweep:

  forward   I_k(T)    the physical device characteristic: how the current of
                      device k responds to temperature.  This is the curve a
                      resistor network would have to reproduce.

  inverse   T(I_k)    the classical single-device calibration curve.  It also
                      serves as the simplest possible baseline in
                      train_regression.py: one device, one polynomial, no
                      network.

Both are fitted on the heating ramp and scored on the cooling curve, so the
reported R^2 already includes the thermal hysteresis of the setup.

Run directly (F5 in Spyder).  Writes figures and a coefficient table into
results/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import data_loader
from poly_basis import format_polynomial

MAX_DEGREE = 5


def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - residual / total if total > 0 else np.nan


def fit_forward_characteristics(data, heating, degrees=range(1, MAX_DEGREE + 1)):
    """
    Fit I_k(T) for every device and every degree.

    Returns (records, coefficients) where `coefficients[(device, degree)]` holds
    the polynomial coefficients in numpy's highest-order-first convention.
    """
    T = data[config.TEMP_COLUMN].values.astype(float)
    records = []
    coefficients = {}

    for col, label in zip(config.CURRENT_COLUMNS, config.DEVICE_LABELS):
        current = data[col].values.astype(float) * 1e9        # work in nA
        for degree in degrees:
            coeffs = np.polyfit(T[heating], current[heating], degree)
            coefficients[(col, degree)] = coeffs
            records.append({
                "device": label,
                "column": col,
                "degree": degree,
                "R2_heating": r_squared(current[heating],
                                        np.polyval(coeffs, T[heating])),
                "R2_cooling": r_squared(current[~heating],
                                        np.polyval(coeffs, T[~heating])),
                "RMSE_cooling_nA": float(np.sqrt(np.mean(
                    (current[~heating] - np.polyval(coeffs, T[~heating])) ** 2))),
            })

    return pd.DataFrame(records), coefficients


def fit_inverse_calibration(data, heating, degree=3):
    """
    Fit T(I_k) for every device: the single-sensor calibration polynomial.

    Currents are scaled to [-1, 1] over the heating range before fitting, so
    the coefficients are well conditioned and directly comparable between
    devices.  Returns a dict keyed by column name.
    """
    T = data[config.TEMP_COLUMN].values.astype(float)
    calibrations = {}

    for col in config.CURRENT_COLUMNS:
        current = data[col].values.astype(float)
        lo, hi = current[heating].min(), current[heating].max()
        span = (hi - lo) or 1.0
        u = 2.0 * (current - lo) / span - 1.0

        coeffs = np.polyfit(u[heating], T[heating], degree)
        calibrations[col] = {
            "coeffs": coeffs,
            "lo": lo,
            "hi": hi,
            "degree": degree,
            "R2_heating": r_squared(T[heating], np.polyval(coeffs, u[heating])),
            "R2_cooling": r_squared(T[~heating], np.polyval(coeffs, u[~heating])),
            "MAE_cooling": float(np.mean(np.abs(
                T[~heating] - np.polyval(coeffs, u[~heating])))),
        }
    return calibrations


def apply_inverse_calibration(calibration, current):
    """Predict temperature from raw current using a fitted calibration."""
    span = (calibration["hi"] - calibration["lo"]) or 1.0
    u = 2.0 * (current - calibration["lo"]) / span - 1.0
    u = np.clip(u, -1.25, 1.25)
    return np.polyval(calibration["coeffs"], u)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_characteristics(data, heating, coefficients, degree=3):
    T = data[config.TEMP_COLUMN].values.astype(float)
    grid = np.linspace(T.min(), T.max(), 300)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Measured device characteristics and degree-%d polynomial fits"
                 % degree, fontsize=14, fontweight="bold")

    for ax, col, label in zip(axes.ravel(), config.CURRENT_COLUMNS,
                              config.DEVICE_LABELS):
        current = data[col].values.astype(float) * 1e9
        ax.scatter(T[heating], current[heating], s=8, alpha=0.45,
                   color="#c0392b", label="heating")
        ax.scatter(T[~heating], current[~heating], s=8, alpha=0.45,
                   color="#2471a3", label="cooling")
        ax.plot(grid, np.polyval(coefficients[(col, degree)], grid), "k-",
                linewidth=2, label="fit on heating")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature [°C]")
        ax.set_ylabel("Current [nA]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(config.result_path("01_device_characteristics.png"),
                dpi=config.DPI, bbox_inches="tight")
    return fig


def plot_degree_scan(table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, group in table.groupby("device", sort=False):
        axes[0].plot(group["degree"], group["R2_heating"], "o-", label=label)
        axes[1].plot(group["degree"], group["R2_cooling"], "o-", label=label)

    axes[0].set_title("Fit quality on the heating ramp (in-sample)",
                      fontsize=12, fontweight="bold")
    axes[1].set_title("Fit quality on the cooling curve (out-of-sample)",
                      fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("R^2")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xticks(sorted(table["degree"].unique()))

    fig.tight_layout()
    fig.savefig(config.result_path("02_polynomial_degree_scan.png"),
                dpi=config.DPI, bbox_inches="tight")
    return fig


def plot_hysteresis(data, heating):
    T = data[config.TEMP_COLUMN].values.astype(float)
    edges = np.arange(30, 156, 5.0)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    for col, label in zip(config.CURRENT_COLUMNS, config.DEVICE_LABELS):
        current = data[col].values.astype(float) * 1e9
        relative = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (T >= lo) & (T < hi)
            up = current[in_bin & heating]
            down = current[in_bin & ~heating]
            if up.size >= 3 and down.size >= 3:
                mean = 0.5 * (up.mean() + down.mean())
                relative.append(100.0 * (up.mean() - down.mean()) / abs(mean))
            else:
                relative.append(np.nan)
        ax.plot(centres, relative, "o-", label=label)

    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("(heating - cooling) / mean  [%]")
    ax.set_title("Thermal hysteresis of the four devices",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(config.result_path("03_hysteresis.png"),
                dpi=config.DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------

def main():
    config.ensure_results_dir()
    data = data_loader.load_raw()
    heating = data_loader.phase_mask(data)

    print("=" * 78)
    print("DEVICE CHARACTERISATION")
    print("=" * 78)
    print("samples: %d   heating: %d   cooling: %d"
          % (len(data), heating.sum(), (~heating).sum()))

    table, coefficients = fit_forward_characteristics(data, heating)
    print("\nForward characteristic  I_k(T)  [current in nA]")
    print("fitted on the heating ramp, scored on the cooling curve\n")
    print(table.to_string(index=False,
                          float_format=lambda v: "%8.4f" % v))
    table.to_csv(config.result_path("device_polynomial_fits.csv"), index=False)

    print("\nDegree-3 characteristics in readable form:")
    for col, label in zip(config.CURRENT_COLUMNS, config.DEVICE_LABELS):
        coeffs = coefficients[(col, 3)][::-1]       # to lowest-order-first
        print("  %-12s I(T) [nA] = %s" % (label, format_polynomial(coeffs, "T", 6)))

    calibrations = fit_inverse_calibration(data, heating, degree=3)
    print("\nInverse calibration  T(I_k)  - single-device baseline")
    print("  %-12s %10s %10s %12s" % ("device", "R2 heat", "R2 cool", "MAE cool"))
    for col, label in zip(config.CURRENT_COLUMNS, config.DEVICE_LABELS):
        cal = calibrations[col]
        print("  %-12s %10.4f %10.4f %10.2f °C"
              % (label, cal["R2_heating"], cal["R2_cooling"], cal["MAE_cooling"]))

    best_degree = 3
    plot_characteristics(data, heating, coefficients, best_degree)
    plot_degree_scan(table)
    plot_hysteresis(data, heating)
    plt.show()

    print("\nFigures written to %s" % config.RESULTS_DIR)
    print("  01_device_characteristics.png")
    print("  02_polynomial_degree_scan.png")
    print("  03_hysteresis.png")
    print("  device_polynomial_fits.csv")
    return table, coefficients, calibrations


if __name__ == "__main__":
    table, coefficients, calibrations = main()
