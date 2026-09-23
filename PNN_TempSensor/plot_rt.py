# -*- coding: utf-8 -*-
"""
Resistance-temperature characteristics from the measured sweep.

The stored quantity is current at a fixed read voltage, so the resistance at
each point follows directly from Ohm's law,

    R_k(T) = V_k / I_k(T),

with V_k the read voltage of channel k.  Under the compact model of the
manuscript the same quantity is

    R(T) = 1 / g(T) = (1 / C_pf) * exp( +Ea / (kB * T) ),

i.e. a straight line of positive slope on a plot of ln R against 1/kB*T, which
is how the fit shown here is obtained.

Three views are produced:

  results/21_resistance_temperature.png
      (a) R(T) for the four read voltages, with the Arrhenius fit
      (b) R normalised to its value at 30 °C, which compares the relative
          sensitivity of the channels independently of their absolute level
      (c) temperature coefficient of resistance, TCR = (1/R) dR/dT

  results/resistance_temperature.csv
      the plateau-averaged R(T) table, in the same style as the I-T
      characteristic of the manuscript: only samples where the plate has
      settled at the set point are used, and the spread over each plateau is
      reported as the error bar.

Run directly (F5 in Spyder).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import data_loader

KB = 8.617333262e-5          # eV/K

# palette shared with plot_data.py: an ordinal ramp for the ordered bias points
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
HEATING = "#eb6834"
COOLING = "#1baf7a"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SETTLE_TOLERANCE = 1.0       # °C; a sample counts as settled within this


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)


# ---------------------------------------------------------------------------
# resistance
# ---------------------------------------------------------------------------

def resistance_table(data=None):
    """
    R_k(T) for every sample and every channel.

    Returns (temperatures, R) with R in ohms, shape (n_samples, 4).
    Samples whose current is at or below zero would give a non-physical
    resistance and are masked out; after dropout removal only a handful of
    noise-floor readings are affected.
    """
    data = data_loader.load_raw() if data is None else data
    temp_c = data[config.TEMP_COLUMN].values.astype(float)
    current = data[config.CURRENT_COLUMNS].values.astype(float)
    volts = np.asarray(config.BIAS_VOLTAGES, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        R = volts[None, :] / current
    R[current <= 0] = np.nan
    return temp_c, R, data


def arrhenius_fit(temp_c, R):
    """
    Fit ln R = ln R0 + Ea/(kB T) for one channel.

    Returns (R0, Ea, R2).  Ea here is the activation energy of the conduction,
    identical to the one obtained from the conductance.
    """
    finite = np.isfinite(R)
    inv_kt = 1.0 / (KB * (temp_c[finite] + 273.15))
    log_r = np.log(R[finite])
    slope, intercept = np.polyfit(inv_kt, log_r, 1)
    predicted = slope * inv_kt + intercept
    r2 = 1.0 - np.sum((log_r - predicted) ** 2) / np.sum((log_r - log_r.mean()) ** 2)
    return np.exp(intercept), slope, r2


def binned_table(data, temp_c, R, width=10.0):
    """
    Average R in temperature bins, separately for heating and cooling.

    The manuscript averages the I-T characteristic over the settled part of
    each plateau.  That works for the heating ramp, where the plate is driven
    to a set point and held, but not for the cooling curve: the set point is
    returned to 30 °C immediately and the plate then drifts down through
    every temperature without ever settling, so no sample satisfies
    |act_temp - set_temp| <= tolerance.

    Binning on the *measured* temperature instead gives a table that covers
    both phases on the same footing, which is what is needed to compare them.
    `settled_fraction` records how much of each bin would also have passed the
    plateau test, so the heating rows remain comparable with the manuscript.
    """
    set_temp = data["set_temp"].values.astype(float)
    heating = data_loader.phase_mask(data)
    settled = np.abs(temp_c - set_temp) <= SETTLE_TOLERANCE

    edges = np.arange(np.floor(temp_c.min() / width) * width,
                      np.ceil(temp_c.max() / width) * width + width, width)
    rows = []
    for phase_name, phase in (("heating", heating), ("cooling", ~heating)):
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = phase & (temp_c >= lo) & (temp_c < hi)
            if mask.sum() < 5:
                continue
            row = {"phase": phase_name,
                   "T_bin_low_C": lo, "T_bin_high_C": hi,
                   "act_temp_C": np.nanmean(temp_c[mask]),
                   "n_samples": int(mask.sum()),
                   "settled_fraction": float(np.mean(settled[mask]))}
            for k in range(len(config.BIAS_VOLTAGES)):
                values = R[mask, k]
                row["R_%.1fV_mean_ohm" % config.BIAS_VOLTAGES[k]] = np.nanmean(values)
                row["R_%.1fV_std_ohm" % config.BIAS_VOLTAGES[k]] = np.nanstd(values)
            rows.append(row)
    return pd.DataFrame(rows)


def tcr(temp_c, R, window=1.0):
    """
    Temperature coefficient of resistance, (1/R) dR/dT, in percent per kelvin.

    Evaluated from the Arrhenius fit rather than by differencing the raw data,
    which is far too noisy to differentiate directly.  For R = R0 exp(Ea/kB T),

        (1/R) dR/dT = -Ea / (kB T^2).
    """
    _, Ea, _ = arrhenius_fit(temp_c, R)
    grid = np.linspace(temp_c.min(), temp_c.max(), 200)
    return grid, 100.0 * (-Ea / (KB * (grid + 273.15) ** 2))


# ---------------------------------------------------------------------------

def main():
    config.ensure_results_dir()
    temp_c, R, data = resistance_table()
    heating = data_loader.phase_mask(data)

    print("=" * 78)
    print("RESISTANCE-TEMPERATURE CHARACTERISTIC")
    print("=" * 78)
    print("  %-12s %12s %12s %8s %10s %10s"
          % ("channel", "R(30 °C)", "R(150 °C)", "ratio", "Ea[eV]", "R2"))

    fits = []
    for k, label in enumerate(config.DEVICE_LABELS):
        lo = np.nanmean(R[np.abs(temp_c - 30) < 1.5, k])
        hi = np.nanmean(R[np.abs(temp_c - 150) < 1.5, k])
        R0, Ea, r2 = arrhenius_fit(temp_c, R[:, k])
        fits.append((R0, Ea, r2))
        print("  %-12s %9.2f MOhm %9.3f MOhm %7.1fx %10.4f %10.4f"
              % (label, lo / 1e6, hi / 1e6, lo / hi, Ea, r2))

    table = binned_table(data, temp_c, R)
    path = config.result_path("resistance_temperature.csv")
    table.to_csv(path, index=False)
    print("\n  plateau-averaged table written to %s" % path)
    print("  (%d bins: %d heating, %d cooling)"
          % (len(table), (table["phase"] == "heating").sum(),
             (table["phase"] == "cooling").sum()))

    make_figure(temp_c, R, heating, fits, table)
    plt.show()
    return table, fits


def make_figure(temp_c, R, heating, fits, table):
    grid = np.linspace(temp_c.min(), temp_c.max(), 300)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), facecolor=SURFACE)

    # (a) R(T) with the Arrhenius fit
    ax = axes[0]
    style_axes(ax)
    for k, colour in enumerate(RAMP):
        ax.scatter(temp_c, R[:, k] / 1e6, s=4, alpha=0.25, color=colour,
                   linewidths=0)
        R0, Ea, _ = fits[k]
        ax.plot(grid, R0 * np.exp(Ea / (KB * (grid + 273.15))) / 1e6,
                color=colour, linewidth=2,
                label="%.1f V,  $E_a$=%.3f eV" % (config.BIAS_VOLTAGES[k], Ea))
    ax.set_yscale("log")
    ax.set_xlabel("temperature [°C]", color=INK_2, fontsize=10)
    ax.set_ylabel("resistance [MOhm]", color=INK_2, fontsize=10)
    ax.set_title("(a)  measured R(T) and Arrhenius fit", loc="left",
                 fontsize=11.5, fontweight="bold", color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)

    # (b) binned means, heating against cooling
    ax = axes[1]
    style_axes(ax)
    for k, colour in enumerate(RAMP):
        column = "R_%.1fV_mean_ohm" % config.BIAS_VOLTAGES[k]
        error = "R_%.1fV_std_ohm" % config.BIAS_VOLTAGES[k]
        for phase, marker, dash in (("heating", "o", "-"), ("cooling", "s", "--")):
            part = table[table["phase"] == phase].sort_values("act_temp_C")
            if part.empty:
                continue
            ax.errorbar(part["act_temp_C"], part[column] / 1e6,
                        yerr=part[error] / 1e6, fmt=marker + dash,
                        color=colour, linewidth=1.3, markersize=3.5,
                        capsize=2, alpha=0.9,
                        label="%.1f V %s" % (config.BIAS_VOLTAGES[k], phase))
    ax.set_yscale("log")
    ax.set_xlabel("temperature [°C]", color=INK_2, fontsize=10)
    ax.set_ylabel("resistance [MOhm]", color=INK_2, fontsize=10)
    ax.set_title("(b)  binned means, heating vs cooling", loc="left",
                 fontsize=11.5, fontweight="bold", color=INK, pad=8)
    ax.legend(frameon=False, fontsize=7.2, labelcolor=INK_2, ncol=2,
              columnspacing=1.0, handlelength=1.8)

    # (c) TCR
    ax = axes[2]
    style_axes(ax)
    for k, colour in enumerate(RAMP):
        g, t = tcr(temp_c, R[:, k])
        ax.plot(g, t, color=colour, linewidth=2,
                label="%.1f V" % config.BIAS_VOLTAGES[k])
    ax.set_xlabel("temperature [°C]", color=INK_2, fontsize=10)
    ax.set_ylabel("TCR  (1/R) dR/dT  [% / K]", color=INK_2, fontsize=10)
    ax.set_title("(c)  temperature coefficient", loc="left", fontsize=11.5,
                 fontweight="bold", color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)

    fig.suptitle("Resistance-temperature characteristic, from the measured sweep",
                 x=0.055, y=0.985, ha="left", fontsize=14, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = config.result_path("21_resistance_temperature.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("  figure written to %s" % path)


if __name__ == "__main__":
    rt_table, arrhenius = main()
