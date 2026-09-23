# -*- coding: utf-8 -*-
"""
One memristor per panel: resistance against temperature, with the polynomial
approximation of equation (1).

------------------------------------------------------------------------------
What is plotted
------------------------------------------------------------------------------
For each device, three things on the same axes:

  measured      R = V_k / I_k, straight from the sweep
  equation (1)  R(T) = (1/C_pf) * exp( +Ea / (kB*T) ), the compact model with
                (C_pf, Ea) fitted to that device
  polynomial    the degree-3 approximation the crossbar actually computes with

------------------------------------------------------------------------------
Which quantity the polynomial approximates
------------------------------------------------------------------------------
The array multiplies and accumulates *conductances*, so the polynomial is
fitted to g(T) and the resistance curve shown here is 1/P(tau).  That is not a
presentational detail: fitting a polynomial straight to R(T) is about eight
times worse over this range, because R is the reciprocal of a near-exponential
and rises steeply at the cold end.

    degree-3, max relative error over 30-150 °C
        polynomial of g, displayed as 1/P     0.22 - 0.33 %
        polynomial fitted directly to R       2.0  - 2.7  %

So the polynomial belongs in conductance, and resistance is the view.

Two figures:

  results/24_device_RT_polynomial.png
      the four measured devices, with residuals underneath

  results/25_array_device_RT.png
      the same picture for the memristors of a trained crossbar, two per
      synapse, showing the states training asks for

Run directly (F5 in Spyder).
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader
import memristor_pnn as mp

MEASURED = "#c3c2b7"
EXACT = "#184f95"
POLY = "#d03b3b"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

N_POINTS = 300
DPI = 1000           # print resolution for the manuscript


def style_axes(ax, grid=True):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.7, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)


def resistance_curves(C_pf, Ea, temp_c, degree=3):
    """
    Exact R(T) from equation (1), and the resistance the degree-D polynomial
    approximation of the conductance produces.
    """
    exact_g = mp.conductance(C_pf, Ea, temp_c)
    poly_g = np.polyval(mp.polynomial_weight(C_pf, Ea, degree),
                        mp.to_tau(temp_c))
    return 1.0 / exact_g, 1.0 / poly_g


# ---------------------------------------------------------------------------
# the four measured devices
# ---------------------------------------------------------------------------

def plot_measured(degree=3):
    data = data_loader.load_raw()
    temp_c = data[config.TEMP_COLUMN].values
    currents = data[config.CURRENT_COLUMNS].values
    states = mp.fit_device_states(data, report=False)
    grid = np.linspace(mp.T_MIN_C, mp.T_MAX_C, N_POINTS)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.9), sharex=True,
                             gridspec_kw={"height_ratios": [2.6, 1.0]},
                             facecolor=SURFACE)

    for k, (column, state) in enumerate(states.items()):
        volt = config.BIAS_VOLTAGES[k]
        measured = np.where(currents[:, k] > 0, volt / currents[:, k], np.nan)
        exact, approx = resistance_curves(state["C_pf"], state["Ea"], grid,
                                          degree)

        ax = axes[0, k]
        style_axes(ax)
        ax.scatter(temp_c, measured / 1e6, s=3, alpha=0.20, color=MEASURED,
                   linewidths=0, label="measured")
        ax.plot(grid, exact / 1e6, color=EXACT, linewidth=2.2,
                label="eq. (1)")
        ax.plot(grid, approx / 1e6, color=POLY, linewidth=1.3,
                linestyle=(0, (4, 3)), label="degree-%d polynomial" % degree)
        ax.set_yscale("log")
        # Signed with the synapse symbol of the drawio schematics.  The
        # argument is tau, the normalised *temperature*, not u: the weight of
        # this model is the device conductance of eq. (1), which is a function
        # of temperature.  P_i(u) with u a normalised current belongs to the
        # signal-domain model of pnn.py, where the weight is not a device.
        coeffs = mp.polynomial_weight(state["C_pf"], state["Ea"],
                                      degree)[::-1] * 1e6
        equation = (r"$P_%d(\tau) = %.3f %+.3f\,\tau %+.3f\,\tau^2 "
                    r"%+.3f\,\tau^3$  $\mu$S"
                    % (k + 1, coeffs[0], coeffs[1], coeffs[2], coeffs[3]))
        ax.set_title(r"$P_%d(\tau)$      %s" % (k + 1, state["label"]),
                     fontsize=11.5, fontweight="bold", color=INK, pad=40)
        ax.text(0.0, 1.105, equation, transform=ax.transAxes,
                fontsize=8.6, color=POLY, va="bottom", ha="left")
        ax.text(0.0, 1.015,
                "$C_{pf}$ = %.2e S,   $E_a$ = %.4f eV"
                % (state["C_pf"], state["Ea"]), transform=ax.transAxes,
                fontsize=8.2, color=INK_2, va="bottom", ha="left")
        if k == 0:
            ax.set_ylabel("resistance [M$\\Omega$]", color=INK_2, fontsize=10)
            ax.legend(frameon=False, fontsize=8.2, labelcolor=INK_2,
                      loc="lower left")

        ax = axes[1, k]
        style_axes(ax)
        residual = 100.0 * (approx - exact) / np.max(exact)
        ax.plot(grid, residual, color=POLY, linewidth=1.6)
        ax.axhline(0, color=AXIS, linewidth=0.9)
        ax.set_xlabel("temperature [°C]", color=INK_2, fontsize=10)
        if k == 0:
            ax.set_ylabel("polynomial\nerror [%]", color=INK_2, fontsize=9.5)
        ax.text(0.97, 0.90, "max %.2f %%" % np.max(np.abs(residual)),
                transform=ax.transAxes, fontsize=8.5, color=POLY,
                ha="right", va="top")

    fig.suptitle("Each memristor: resistance against temperature, and the "
                 "polynomial that stands in for equation (1)",
                 x=0.045, y=0.985, ha="left", fontsize=14,
                 fontweight="bold", color=INK)
    fig.text(0.045, 0.932,
             r"$P_i(\tau)$ is the synapse of the schematic: the degree-3 "
             r"polynomial standing in for eq. (1), with $\tau$ the temperature "
             r"scaled to $[-1,1]$ over 30-150 °C." "\n"
             r"It is fitted to the $conductance$, because that is what the "
             r"array computes with, so the curve drawn here is "
             r"$1/P_i(\tau)$ - fitting a polynomial straight to $R(T)$ is "
             r"about eight times worse.",
             ha="left", va="top", fontsize=9.2, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = config.result_path("24_device_RT_polynomial.png")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    print("written: %s" % path)
    return states


# ---------------------------------------------------------------------------
# the memristors of a trained array
# ---------------------------------------------------------------------------

def plot_array(n_classes=6, seed=0, degree=3, max_cells=6):
    ds = data_loader.prepare(split="ramp", n_classes=n_classes)
    x_read = np.asarray(config.BIAS_VOLTAGES, float)
    ea_range, c_range = mp.measured_envelope()
    crossbar = mp.ArrheniusCrossbar(4, n_classes, ea_range=ea_range,
                                    c_range=c_range, degree=degree, seed=seed)
    mp.train_crossbar(crossbar, x_read, ds.T_train, ds.y_class_train,
                      ds.T_val, ds.y_class_val, epochs=2000, lr=0.02,
                      verbose=0, recalibrate_every=250)

    grid = np.linspace(mp.T_MIN_C, mp.T_MAX_C, N_POINTS)
    cells = [(i, j) for i in range(crossbar.n_in)
             for j in range(crossbar.n_out)][:max_cells]

    fig, axes = plt.subplots(1, len(cells), figsize=(2.5 * len(cells), 3.9),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)

    for ax, (i, j) in zip(axes, cells):
        style_axes(ax)
        for coeff, energy, colour, name in (
                (np.exp(crossbar.c_p[i, j]), crossbar.Ea_p[i, j], "#c0392b", "A"),
                (np.exp(crossbar.c_n[i, j]), crossbar.Ea_n[i, j], "#2980b9", "B")):
            exact, approx = resistance_curves(coeff, energy, grid, degree)
            ax.plot(grid, exact / 1e6, color=colour, linewidth=2,
                    label="%s: $E_a$=%.3f eV" % (name, energy))
            ax.plot(grid, approx / 1e6, color="black", linewidth=0.8,
                    linestyle=(0, (3, 2.5)))
        ax.set_yscale("log")
        ax.set_xlabel("temperature [°C]", color=INK_2, fontsize=9)
        ax.set_title("D%d @ %.1f V -> %s °C"
                     % (i + 1, config.BIAS_VOLTAGES[i],
                        ds.class_names_short[j]),
                     fontsize=9, fontweight="bold", color=INK, pad=6)
        ax.legend(frameon=False, fontsize=7.2, labelcolor=INK_2,
                  loc="upper right")
    axes[0].set_ylabel("resistance [M$\\Omega$]", color=INK_2, fontsize=10)

    fig.suptitle("Memristors of a trained %d x %d array: the two devices of "
                 "each synapse, in resistance"
                 % (crossbar.n_in, crossbar.n_out),
                 x=0.03, y=0.995, ha="left", fontsize=12.5,
                 fontweight="bold", color=INK)
    fig.text(0.03, 0.945,
             "Solid: equation (1) at the programmed state training asks for.  "
             "Dashed black: its degree-%d polynomial." % degree,
             ha="left", va="top", fontsize=8.6, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = config.result_path("25_array_device_RT.png")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    print("written: %s" % path)
    return crossbar


def main():
    config.ensure_results_dir()
    print("=" * 78)
    print("RESISTANCE-TEMPERATURE PER MEMRISTOR, WITH THE POLYNOMIAL")
    print("=" * 78)

    print("\n  Degree-3 accuracy, maximum relative error over 30-150 °C:")
    print("  %-10s %24s %24s"
          % ("Ea [eV]", "poly of g, shown as R", "poly fitted to R"))
    temp_c = np.linspace(mp.T_MIN_C, mp.T_MAX_C, 400)
    for Ea in (0.195, 0.203, 0.215, 0.224):
        exact = 1.0 / mp.conductance(1.0, Ea, temp_c)
        from_g = 1.0 / np.polyval(mp.polynomial_weight(1.0, Ea, 3),
                                  mp.to_tau(temp_c))
        direct = np.polyval(np.polyfit(mp.to_tau(temp_c), exact, 3),
                            mp.to_tau(temp_c))
        print("  %-10.3f %23.4f%% %23.4f%%"
              % (Ea,
                 100 * np.max(np.abs(from_g - exact)) / np.max(exact),
                 100 * np.max(np.abs(direct - exact)) / np.max(exact)))

    print()
    states = plot_measured()
    crossbar = plot_array()
    plt.show()
    return states, crossbar


if __name__ == "__main__":
    device_states, trained_array = main()
