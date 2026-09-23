# -*- coding: utf-8 -*-
"""
Generate the figures for the "Physical neural network design" section, sized
for IEEEtran two-column layout.

    img/pnnArchitecture.png   double column (figure*), 7.16 in
    img/pnnWeights.png        single column, 3.5 in
    img/pnnDegree.png         single column, 3.5 in

Everything is rendered at 600 dpi for print.  Run this from the paper_pnn
folder; it imports the PNN package from ../PNN_TempSensor.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.join(os.path.dirname(HERE), "PNN_TempSensor")
sys.path.insert(0, PACKAGE)

import config                                    # noqa: E402
import data_loader                               # noqa: E402
from poly_basis import basis_matrices            # noqa: E402
from pnn import PNN                              # noqa: E402

IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)

# Single-hue ordinal ramp for the four read voltages (ordered quantity).
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
REFERENCE = "#898781"
INK = "#0b0b0b"
INK_2 = "#3d3d3a"
GRID = "#d8d7d0"
AXIS = "#9a9992"

COL_1 = 3.50          # IEEE single column, inches
COL_2 = 7.16          # IEEE double column, inches
DPI = 1000

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def style(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, width=0.8, length=3)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def train(ds, degree, hidden=(), seed=0):
    net = PNN(n_in=4, n_out=1, hidden=hidden, degree=degree,
              basis=config.POLY_BASIS, task="regression", seed=seed)
    net.fit(ds.X_train, ds.y_reg_train, ds.X_val, ds.y_reg_val,
            epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE, l2=config.L2, clip=config.GRAD_CLIP,
            patience=config.PATIENCE, seed=seed, verbose=0)
    return net


def test_mae(net, ds):
    pred = ds.target_scaler.inverse_transform(net.predict(ds.X_test).ravel())
    return float(np.mean(np.abs(pred - ds.T_test)))


# ---------------------------------------------------------------------------
# Fig. 1 - architecture comparison (double column)
# ---------------------------------------------------------------------------

def draw_node(ax, x, y, r, label, facecolor="white", edgecolor=INK,
              fontsize=7.5, weight="normal"):
    ax.add_patch(Circle((x, y), r, facecolor=facecolor, edgecolor=edgecolor,
                        linewidth=0.9, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=4, fontweight=weight)


def edge(ax, x0, y0, x1, y1, color=REFERENCE, lw=0.7, alpha=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-",
                                 color=color, linewidth=lw, alpha=alpha,
                                 zorder=1, shrinkA=0, shrinkB=0))


def plot_architecture(readout_coeffs=None):
    """
    The two arrangements, with the notation of the schematics.

    Panel (a) senses the currents and computes in a separate network, so its
    inputs are I(V_i) and its synapses are scalars.  Panel (b) computes in the
    array itself: the inputs are the applied read voltages V_i and the synapse
    is the device characteristic P_i(tau), a polynomial in normalised
    *temperature*.  The two panels therefore take different inputs, and that
    difference is the point of the comparison rather than an inconsistency.
    """
    import memristor_pnn as mp

    states = mp.fit_device_states(report=False)
    # the real g(tau) of each measured device, for the curves inside the boxes
    device_curves = [mp.polynomial_weight(s["C_pf"], s["Ea"], 3)[::-1]
                     for s in states.values()]

    fig, axes = plt.subplots(1, 2, figsize=(COL_2, 2.95))
    labels_a = [r"$I(V_1)$", r"$I(V_2)$", r"$I(V_3)$", r"$I(V_4)$"]
    labels_b = [r"$V_1$", r"$V_2$", r"$V_3$", r"$V_4$"]
    labels = labels_a
    volts = ["1.2 V", "1.4 V", "1.6 V", "1.8 V"]
    y_in = np.linspace(0.80, 0.20, 4)
    r = 0.052

    # ---- (a) conventional network ------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 1.0)
    ax.axis("off")

    y_hidden = np.linspace(0.86, 0.14, 5)
    for i, yi in enumerate(y_in):
        for yh in y_hidden:
            edge(ax, 0.20 + r, yi, 0.55 - r, yh, lw=0.5, alpha=0.55)
    for yh in y_hidden:
        edge(ax, 0.55 + r, yh, 0.87 - r, 0.5, lw=0.5, alpha=0.55)

    for i, (yi, lab) in enumerate(zip(y_in, labels)):
        draw_node(ax, 0.20, yi, r, lab, facecolor="white", edgecolor=RAMP[i])
        ax.text(0.20 - r - 0.02, yi, volts[i], ha="right", va="center",
                fontsize=6.8, color=INK_2)
    for k, yh in enumerate(y_hidden):
        lab = r"$\tanh$" if k != 2 else r"$\vdots$"
        draw_node(ax, 0.55, yh, r, lab, facecolor="#f2f1ec",
                  edgecolor=REFERENCE, fontsize=6.5)
    draw_node(ax, 0.87, 0.5, r, r"$\hat{T}$", facecolor="white",
              edgecolor=INK, weight="bold")

    ax.text(0.5, 0.985, "(a)  conventional neural network",
            ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    ax.text(0.5, 0.055, r"senses $I(V_i)$, computes in a separate network"
                        "\n" r"scalar $w_{ij}$ + hidden layer, 161 parameters",
            ha="center", va="top", fontsize=7.4, color=INK_2)

    # ---- (b) polynomial crossbar --------------------------------------
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 1.0)
    ax.axis("off")

    u = np.linspace(-1, 1, 80)
    for i, yi in enumerate(y_in):
        edge(ax, 0.17 + r, yi, 0.80 - r, 0.5, color=RAMP[i], lw=1.0, alpha=0.9)
        # the synapse itself: a miniature plot of P_i(u) sitting on the wire
        curve = np.polyval(device_curves[i][::-1], u)
        span = np.ptp(curve) or 1.0
        # centre the synapse box on the wire it belongs to
        # Sit the synapse box on its own wire, close to the input where the
        # wires are still well separated.  The curve is colour-matched to its
        # input node, so no per-box label is needed.
        x0, x1 = 0.17 + r, 0.80 - r
        w, h = 0.150, 0.048
        cx = 0.355
        cy = yi + (cx - x0) / (x1 - x0) * (0.5 - yi)
        ax.add_patch(plt.Rectangle((cx - w / 2 - 0.010, cy - h),
                                   w + 0.020, 2 * h, facecolor="white",
                                   edgecolor=GRID, linewidth=0.6, zorder=4))
        ax.plot(cx - w / 2 + w * (u + 1) / 2,
                cy + 0.80 * h * (curve - curve.mean()) / span,
                color=RAMP[i], linewidth=1.3, zorder=5)

    for i, (yi, lab) in enumerate(zip(y_in, labels_b)):
        draw_node(ax, 0.17, yi, r, lab, facecolor="white", edgecolor=RAMP[i])
        ax.text(0.17 - r - 0.02, yi, volts[i], ha="right", va="center",
                fontsize=6.8, color=INK_2)
    draw_node(ax, 0.80, 0.5, r, r"$\hat{T}$", facecolor="white",
              edgecolor=INK, weight="bold")
    ax.annotate(r"$P_i(\tau)$", xy=(0.355, 0.80 + 0.048 + 0.004),
                xytext=(0.50, 0.90), fontsize=7.4, color=RAMP[0],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RAMP[0], linewidth=0.7))

    ax.text(0.5, 0.985, "(b)  polynomial crossbar (this work)",
            ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    ax.text(0.5, 0.055, r"applies $V_i$, the synapse $P_i(\tau)$ is the device"
                        "\n" r"$\tau$ = normalised temperature,  no hidden layer",
            ha="center", va="top", fontsize=7.4, color=INK_2)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.05)
    path = os.path.join(IMG, "pnnArchitecture.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)


# ---------------------------------------------------------------------------
# Fig. 2 - the learned synapses (single column)
# ---------------------------------------------------------------------------

def plot_weights(net_poly, net_linear):
    layer = net_poly.poly_layers()[0]
    u = np.linspace(-1, 1, 240)
    Phi, _ = basis_matrices(u.reshape(-1, 1), layer.degree, layer.basis)
    curves = np.einsum("np,iop->ion", Phi[:, 0, :], layer.params["C"])

    lin = net_linear.poly_layers()[0]
    Phi1, _ = basis_matrices(u.reshape(-1, 1), lin.degree, lin.basis)
    flat = np.einsum("np,iop->ion", Phi1[:, 0, :], lin.params["C"])

    fig, ax = plt.subplots(figsize=(COL_1, 2.55))
    style(ax)
    for i in range(4):
        ax.plot(u, flat[i, 0], color=REFERENCE, linewidth=0.8,
                linestyle=(0, (4, 2.5)), zorder=2)
        ax.plot(u, curves[i, 0], color=RAMP[i], linewidth=1.7, zorder=3,
                label=r"$P_%d$  (%.1f V)" % (i + 1, config.BIAS_VOLTAGES[i]))
    ax.plot([], [], color=REFERENCE, linewidth=0.8, linestyle=(0, (4, 2.5)),
            label="degree 1")
    ax.axhline(0, color=AXIS, linewidth=0.7, zorder=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"normalised read current $u$")
    ax.set_ylabel(r"synapse output $P(u)$")
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), columnspacing=1.0,
              handlelength=1.8, labelspacing=0.25)
    ax.set_ylim(bottom=ax.get_ylim()[0] - 1.4)

    fig.tight_layout(pad=0.25)
    path = os.path.join(IMG, "pnnWeights.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)


# ---------------------------------------------------------------------------
# Fig. 3 - degree scan (single column)
# ---------------------------------------------------------------------------

def plot_degree_scan(ds, degrees=(1, 2, 3, 4, 5), seeds=(0, 1, 2)):
    train_mae, test_mae, test_sd = [], [], []
    for degree in degrees:
        runs = []
        for seed in seeds:
            net = train(ds, degree, seed=seed)
            tr = ds.target_scaler.inverse_transform(net.predict(ds.X_train).ravel())
            runs.append((np.mean(np.abs(tr - ds.T_train)), test_mae_of(net, ds)))
        runs = np.array(runs)
        train_mae.append(runs[:, 0].mean())
        test_mae.append(runs[:, 1].mean())
        test_sd.append(runs[:, 1].std())
        print("  degree %d: train %.2f  test %.2f +- %.2f"
              % (degree, train_mae[-1], test_mae[-1], test_sd[-1]))

    fig, ax = plt.subplots(figsize=(COL_1, 2.45))
    style(ax)
    ax.plot(degrees, train_mae, "o-", color="#2980b9", linewidth=1.4,
            markersize=4, label="training set (heating)")
    ax.errorbar(degrees, test_mae, yerr=test_sd, fmt="s-", color="#c0392b",
                linewidth=1.4, markersize=4, capsize=2.5,
                label="test set (cooling)")
    ax.set_xticks(list(degrees))
    ax.set_xlabel("degree of the synaptic polynomial")
    ax.set_ylabel(r"temperature MAE [°C]")
    ax.legend(frameon=False, loc="upper center", handlelength=1.8)
    ax.set_ylim(0, max(test_mae) * 1.30)
    ax.annotate("degree 1 =\nconventional\ncrossbar", xy=(1, test_mae[0]),
                xytext=(1.35, test_mae[0] * 0.62), fontsize=6.8, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_2, linewidth=0.7))

    fig.tight_layout(pad=0.25)
    path = os.path.join(IMG, "pnnDegree.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)
    return train_mae, test_mae, test_sd


def test_mae_of(net, ds):
    pred = ds.target_scaler.inverse_transform(net.predict(ds.X_test).ravel())
    return float(np.mean(np.abs(pred - ds.T_test)))


# ---------------------------------------------------------------------------

def main():
    ds = data_loader.prepare(split="ramp")
    print("train %d / val %d / test %d"
          % (len(ds.idx_train), len(ds.idx_val), len(ds.idx_test)))

    net_poly = train(ds, config.POLY_DEGREE)
    net_linear = train(ds, 1)
    net_ann = train(ds, 1, hidden=config.BASELINE_HIDDEN)

    print("\nPNN crossbar      MAE %.2f °C  (%d parameters)"
          % (test_mae_of(net_poly, ds), net_poly.n_parameters()))
    print("linear crossbar   MAE %.2f °C  (%d parameters)"
          % (test_mae_of(net_linear, ds), net_linear.n_parameters()))
    print("conventional ANN  MAE %.2f °C  (%d parameters)"
          % (test_mae_of(net_ann, ds), net_ann.n_parameters()))

    power = net_poly.poly_layers()[0].power_coefficients()[:, 0, :]
    print("\nLaTeX table rows (a0 .. a3):")
    for i in range(4):
        print("  %.1f & %s \\\\" % (config.BIAS_VOLTAGES[i],
                                    " & ".join("%+.3f" % c for c in power[i])))

    plot_architecture(power)
    plot_behaviour()
    plot_weights(net_poly, net_linear)
    print("\ndegree scan:")
    plot_degree_scan(ds)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Fig. 4 - the memristive weight behaviour (double column)
# ---------------------------------------------------------------------------

def plot_behaviour():
    """
    What the weight actually does, now that it is the device conductance.

      (a) g(T) of equation (2) for the four measured states, with the
          degree-3 polynomial surrogate laid over it
      (b) how close that surrogate is, against activation energy and degree
      (c) trained differential weights, which are non-monotonic in T and
          therefore band-selective
    """
    import memristor_pnn as mp

    states = mp.fit_device_states(report=False)
    temp_c = np.linspace(mp.T_MIN_C, mp.T_MAX_C, 300)

    fig, axes = plt.subplots(1, 3, figsize=(COL_2, 2.35))

    # (a) equation (2) and its polynomial surrogate
    ax = axes[0]
    style(ax)
    for (column, st), colour in zip(states.items(), RAMP):
        ax.semilogy(temp_c, mp.conductance(st["C_pf"], st["Ea"], temp_c),
                    color=colour, linewidth=1.6,
                    label=r"$E_a$=%.3f eV" % st["Ea"])
        ax.semilogy(temp_c,
                    np.polyval(mp.polynomial_weight(st["C_pf"], st["Ea"], 3),
                               mp.to_tau(temp_c)),
                    color="black", linewidth=0.7, linestyle=(0, (3, 2.5)))
    ax.plot([], [], color="black", linewidth=0.7, linestyle=(0, (3, 2.5)),
            label="degree-3 fit")
    ax.set_xlabel(r"temperature [°C]")
    ax.set_ylabel(r"weight $g(T)$ [S]")
    ax.set_title("(a) the weight is the conductance", fontsize=8.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right",
              handlelength=1.5, labelspacing=0.22)

    # (b) how good the surrogate is
    ax = axes[1]
    style(ax)
    energies = np.linspace(0.10, 0.45, 60)
    for degree, colour in zip((1, 2, 3, 4),
                              ["#c0392b", "#e67e22", "#27ae60", "#2980b9"]):
        ax.semilogy(energies,
                    [mp.approximation_error(1.0, Ea, degree) for Ea in energies],
                    color=colour, linewidth=1.5, label="$D=%d$" % degree)
    ax.axvspan(0.195, 0.224, color="#95a5a6", alpha=0.30, linewidth=0)
    ax.text(0.2095, 28, "measured", ha="center", fontsize=6.5, color=INK_2)
    ax.set_xlabel(r"activation energy $E_a$ [eV]")
    ax.set_ylabel("max. error [%]")
    ax.set_title("(b) cost of the polynomial", fontsize=8.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right",
              handlelength=1.5, labelspacing=0.22, columnspacing=0.9)

    # (c) trained differential weights
    ax = axes[2]
    style(ax)
    ds = data_loader.prepare(split="ramp", n_classes=config.N_CLASSES)
    x_read = np.asarray(config.BIAS_VOLTAGES, float)
    crossbar = mp.ArrheniusCrossbar(4, ds.n_classes, ea_range=(0.10, 0.45),
                                    c_range=(1e-5, 1e-1), degree=3, seed=0)
    mp.train_crossbar(crossbar, x_read, ds.T_train, ds.y_class_train,
                      ds.T_val, ds.y_class_val, epochs=2000, lr=0.02,
                      verbose=0, recalibrate_every=250)
    w = crossbar.weights(temp_c, mode="exact")
    shown = [0, 3, 6, 9, 11]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(shown)))
    for k, j in enumerate(shown):
        ax.plot(temp_c, w[:, 0, j] * 1e6, color=cmap[k], linewidth=1.5,
                label=r"%.0f$-$%.0f" % (ds.class_edges[j], ds.class_edges[j + 1]))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel(r"temperature [°C]")
    ax.set_ylabel(r"$w_{1j}(T)$ [$\mu$S]")
    ax.set_title("(c) trained weights, physical states", fontsize=8.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left",
              handlelength=1.4, labelspacing=0.22, columnspacing=0.8,
              title=r"band [°C]", title_fontsize=6.2)

    fig.tight_layout(pad=0.3, w_pad=1.1)
    path = os.path.join(IMG, "pnnBehaviour.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)


# ---------------------------------------------------------------------------
# Resistance-temperature figures (single column each)
# ---------------------------------------------------------------------------

def plot_resistance():
    """
    Three single-column figures on the resistance of the measured devices.

        img/rtCharacteristic.png   R(T) with the Arrhenius fit
        img/rtTCR.png              temperature coefficient of resistance
        img/rtHysteresis.png       cooling/heating ratio, which isolates the
                                   thermal hysteresis from the R(T) trend
    """
    import plot_rt as rt

    temp_c, R, data = rt.resistance_table()
    table = rt.binned_table(data, temp_c, R)
    grid = np.linspace(temp_c.min(), temp_c.max(), 300)
    volts = config.BIAS_VOLTAGES

    # ---- (1) R(T) and the Arrhenius fit -------------------------------
    fig, ax = plt.subplots(figsize=(COL_1, 2.7))
    style(ax)
    fits = []
    for k, colour in enumerate(RAMP):
        R0, Ea, r2 = rt.arrhenius_fit(temp_c, R[:, k])
        fits.append((R0, Ea, r2))
        ax.scatter(temp_c, R[:, k] / 1e6, s=2.0, alpha=0.18, color=colour,
                   linewidths=0)
        ax.plot(grid, R0 * np.exp(Ea / (rt.KB * (grid + 273.15))) / 1e6,
                color=colour, linewidth=1.5,
                label=r"%.1f V,  $E_a$ = %.3f eV" % (volts[k], Ea))
    ax.set_yscale("log")
    ax.set_xlabel(r"temperature [°C]")
    ax.set_ylabel(r"resistance [M$\Omega$]")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right",
              handlelength=1.6, labelspacing=0.28)
    fig.tight_layout(pad=0.25)
    path = os.path.join(IMG, "rtCharacteristic.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)

    # ---- (2) TCR -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(COL_1, 2.5))
    style(ax)
    for k, colour in enumerate(RAMP):
        Ea = fits[k][1]
        ax.plot(grid, 100.0 * (-Ea / (rt.KB * (grid + 273.15) ** 2)),
                color=colour, linewidth=1.6, label="%.1f V" % volts[k])
    ax.set_xlabel(r"temperature [°C]")
    # "%" is a plain character here: this text is outside math mode, and
    # escaping it as \% would render the backslash literally.
    ax.set_ylabel(r"TCR  $R^{-1}\,\mathrm{d}R/\mathrm{d}T$  [% / K]")
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="lower right",
              handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(pad=0.25)
    path = os.path.join(IMG, "rtTCR.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)

    # ---- (3) hysteresis as a ratio ------------------------------------
    # Plotting cooling/heating rather than two R(T) curves removes the strong
    # common trend and leaves only the effect being described, which is a few
    # tens of percent and would otherwise be invisible on a log axis.
    fig, ax = plt.subplots(figsize=(COL_1, 2.5))
    style(ax)
    heat = table[table["phase"] == "heating"].sort_values("act_temp_C")
    cool = table[table["phase"] == "cooling"].sort_values("act_temp_C")
    for k, colour in enumerate(RAMP):
        column = "R_%.1fV_mean_ohm" % volts[k]
        common = np.intersect1d(heat["T_bin_low_C"], cool["T_bin_low_C"])
        h = heat[heat["T_bin_low_C"].isin(common)]
        c = cool[cool["T_bin_low_C"].isin(common)]
        ratio = c[column].values / h[column].values
        ax.plot(0.5 * (h["act_temp_C"].values + c["act_temp_C"].values), ratio,
                "o-", color=colour, linewidth=1.5, markersize=3.5,
                label="%.1f V" % volts[k])
    ax.axhline(1.0, color=AXIS, linewidth=0.9)
    ax.set_xlabel(r"temperature [°C]")
    ax.set_ylabel(r"$R_\mathrm{cooling}\,/\,R_\mathrm{heating}$")
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper right",
              handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(pad=0.25)
    path = os.path.join(IMG, "rtHysteresis.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)

    print("\nArrhenius parameters used in the figures:")
    for k, (R0, Ea, r2) in enumerate(fits):
        print("  %.1f V:  R0 = %.3e Ohm   Ea = %.4f eV   R2 = %.4f"
              % (volts[k], R0, Ea, r2))
    return fits
