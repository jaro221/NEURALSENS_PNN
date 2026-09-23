# -*- coding: utf-8 -*-
"""
Plot the polynomial weights as line plots, each labelled with its exact
polynomial equation.

  results/18_polynomial_weights.png
      the 4 x 1 temperature read-out crossbar: one panel per device, the
      learned characteristic P(u) drawn as a line and its exact equation
      printed above the panel.  The degree-1 weight the same crossbar learns
      is drawn alongside, with its equation, so the two are comparable.

  results/19_weight_matrix.png
      the full 4 x 12 classification crossbar: 48 synapses, each a line plot
      with its equation.

  results/polynomial_weights.txt
      every equation in full precision, for copying into a report.

u on every x-axis is the normalised device current, in [-1, 1]: -1 is the
lowest current that device produced on the training data, +1 the highest.
The coefficients are printed in the ordinary power basis, so

    P(u) = a0 + a1*u + a2*u^2 + a3*u^3

reads directly even though training runs in the Chebyshev basis.

Run directly (F5 in Spyder).
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader
from poly_basis import basis_matrices
from pnn import PNN

# --- palette (shared with plot_data.py) ------------------------------------
DEVICE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
REFERENCE = "#898781"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

N_POINTS = 240


def style_axes(ax, grid=True):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3.5, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)


# ---------------------------------------------------------------------------
# equations
# ---------------------------------------------------------------------------

def equation_mathtext(coeffs, decimals=3, var="u", lhs="P(u)", start_order=0):
    """
    Render power-basis coefficients as mathtext, e.g.

        $P(u) = 0.654 - 0.025\\,u + 0.005\\,u^2 + 0.061\\,u^3$

    Every term is kept, including ones that round to zero, so the degree of
    the polynomial stays visible.  `start_order` lets a long equation be split
    across two lines without renumbering the powers.
    """
    parts = []
    for k, a in enumerate(np.atleast_1d(coeffs)):
        p = k + start_order
        value = float(a)
        sign = "-" if value < 0 else "+"
        magnitude = "%.*f" % (decimals, abs(value))
        if k == 0 and start_order == 0:
            term = ("-" if value < 0 else "") + magnitude
            if p == 1:
                term += r"\,%s" % var
            elif p > 1:
                term += r"\,%s^{%d}" % (var, p)
        elif p == 0:
            term = r"%s\,%s" % (sign, magnitude)
        elif p == 1:
            term = r"%s\,%s\,%s" % (sign, magnitude, var)
        else:
            term = r"%s\,%s\,%s^{%d}" % (sign, magnitude, var, p)
        parts.append(term)
    body = " ".join(parts)
    return r"$%s = %s$" % (lhs, body) if lhs else r"$%s$" % body


def equation_text(coeffs, decimals=6, var="u", lhs="P(u)"):
    """Plain-text version for the console and the .txt file."""
    parts = []
    for p, a in enumerate(np.atleast_1d(coeffs)):
        value = float(a)
        sign = "- " if value < 0 else "+ "
        magnitude = "%.*f" % (decimals, abs(value))
        if p == 0:
            parts.append(("-" if value < 0 else "") + magnitude)
        elif p == 1:
            parts.append("%s%s*%s" % (sign, magnitude, var))
        else:
            parts.append("%s%s*%s^%d" % (sign, magnitude, var, p))
    return "%s = %s" % (lhs, " ".join(parts))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(ds, n_out, task, degree, seed=0):
    net = PNN(n_in=ds.X_train.shape[1], n_out=n_out, hidden=(), degree=degree,
              basis=config.POLY_BASIS, task=task, seed=seed)
    target_train = ds.y_reg_train if task == "regression" else ds.y_class_train
    target_val = ds.y_reg_val if task == "regression" else ds.y_class_val
    net.fit(ds.X_train, target_train, ds.X_val, target_val,
            epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE, l2=config.L2, clip=config.GRAD_CLIP,
            patience=config.PATIENCE, seed=seed, verbose=0)
    return net


def curves(net, layer_index=0):
    layer = net.poly_layers()[layer_index]
    u = np.linspace(-1.0, 1.0, N_POINTS)
    Phi, _ = basis_matrices(u.reshape(-1, 1), layer.degree, layer.basis)
    return u, np.einsum("np,iop->ion", Phi[:, 0, :], layer.params["C"])


# ---------------------------------------------------------------------------
# figure 1 - the read-out crossbar, one panel per device
# ---------------------------------------------------------------------------

def plot_readout_weights(net_poly, net_linear):
    u, poly_curves = curves(net_poly)
    _, flat_curves = curves(net_linear)
    poly_power = net_poly.poly_layers()[0].power_coefficients()
    flat_power = net_linear.poly_layers()[0].power_coefficients()

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.780, bottom=0.070,
                        wspace=0.20, hspace=0.62)

    for i, ax in enumerate(axes.ravel()):
        style_axes(ax)
        ax.plot(u, flat_curves[i, 0], color=REFERENCE, linewidth=1.8,
                linestyle=(0, (5, 3)), zorder=2, label="degree 1")
        ax.plot(u, poly_curves[i, 0], color=DEVICE_RAMP[i], linewidth=2.6,
                zorder=3, label="degree %d" % net_poly.degree)
        ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)

        ax.set_xlim(-1, 1)
        ax.set_xlabel("normalised device current  u", color=INK_2, fontsize=9.5)
        ax.set_ylabel("synapse output  P(u)", color=INK_2, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=2,
                  loc="best", handlelength=2.4)

        # Device name and the two equations as three stacked header lines, so
        # nothing can collide with a left-aligned title.
        ax.text(0.0, 1.275, "D%d  @  %.1f V"
                % (i + 1, config.BIAS_VOLTAGES[i]), transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=INK,
                va="bottom", ha="left")
        ax.text(0.0, 1.145, equation_mathtext(poly_power[i, 0], 3),
                transform=ax.transAxes, fontsize=11,
                color=DEVICE_RAMP[i], va="bottom", ha="left")
        ax.text(0.0, 1.030, equation_mathtext(flat_power[i, 0], 3),
                transform=ax.transAxes, fontsize=10,
                color=REFERENCE, va="bottom", ha="left")

    fig.suptitle("Polynomial weights of the 4 x 1 temperature read-out crossbar",
                 x=0.075, y=0.985, ha="left", fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.075, 0.950,
             "One synapse per device.  The coloured line and its equation are "
             "the learned degree-%d characteristic; the grey line is the "
             "straight-line weight\nthe same crossbar is left with at degree 1 "
             "(23.8 °C test error, against 2.5 °C for the curves).  "
             "u is the normalised device current."
             % net_poly.degree,
             ha="left", va="top", fontsize=9.8, color=INK_2)

    path = config.result_path("18_polynomial_weights.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("written: %s" % path)
    return poly_power, flat_power


# ---------------------------------------------------------------------------
# figure 2 - the full weight matrix, every synapse a line plot
# ---------------------------------------------------------------------------

def plot_weight_matrix(net, class_names):
    u, matrix = curves(net)
    layer = net.poly_layers()[0]
    power = layer.power_coefficients()
    n_in, n_out = layer.n_in, layer.n_out

    lo, hi = matrix.min() * 1.15, matrix.max() * 1.15

    fig, axes = plt.subplots(n_in, n_out,
                             figsize=(2.10 * n_out + 1.4, 2.35 * n_in + 1.6),
                             sharex=True, sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.050, right=0.995, top=0.825, bottom=0.070,
                        wspace=0.16, hspace=0.56)

    for i in range(n_in):
        for j in range(n_out):
            ax = axes[i, j]
            style_axes(ax, grid=False)
            ax.axhline(0, color=GRID, linewidth=1.0, zorder=1)
            ax.plot(u, matrix[i, j], color=DEVICE_RAMP[i], linewidth=2.0,
                    zorder=3)
            ax.set_ylim(lo, hi)
            ax.set_xlim(-1, 1)
            ax.set_xticks([-1, 0, 1])

            # equation split over two lines so it fits the cell
            coeffs = power[i, j]
            ax.text(0.5, 1.115, equation_mathtext(coeffs[:2], 2),
                    transform=ax.transAxes, fontsize=7.8, color=INK_2,
                    ha="center", va="bottom")
            ax.text(0.5, 1.020,
                    equation_mathtext(coeffs[2:], 2, lhs="", start_order=2),
                    transform=ax.transAxes, fontsize=7.8, color=INK_2,
                    ha="center", va="bottom")

            if i == 0:
                ax.set_title("%s °C" % class_names[j], fontsize=9.5,
                             color=INK, fontweight="bold", pad=34)
            if j == 0:
                ax.set_ylabel("D%d\n%.1f V" % (i + 1, config.BIAS_VOLTAGES[i]),
                              fontsize=9.5, color=INK, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=20)
            else:
                ax.tick_params(labelleft=False)
            if i < n_in - 1:
                ax.tick_params(labelbottom=False)

    fig.suptitle("The PNN weight matrix: every entry is a polynomial, not a number",
                 x=0.050, y=0.975, ha="left", fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.050, 0.945,
             "All %d synapses of the 4 x %d classification crossbar, each drawn "
             "as a line with its exact equation.  Rows are devices, columns are "
             "temperature bands.\nx-axis: normalised device current u in "
             "[-1, 1].  All panels share one y-scale, so curve heights are "
             "comparable across the whole matrix."
             % (n_in * n_out, n_out),
             ha="left", va="top", fontsize=9.8, color=INK_2)
    fig.text(0.5, 0.020, "normalised device current  u", ha="center",
             fontsize=10.5, color=INK_2)

    path = config.result_path("19_weight_matrix.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("written: %s" % path)
    return power


# ---------------------------------------------------------------------------

def write_equations(readout_power, flat_power, matrix_power, class_names):
    degree = readout_power.shape[-1] - 1
    lines = ["POLYNOMIAL WEIGHTS - exact coefficients",
             "=" * 74,
             "",
             "u is the normalised device current in [-1, 1]:",
             "  u = -1 is the lowest current that device produced in training,",
             "  u = +1 the highest.",
             "",
             "-" * 74,
             "4 x 1 temperature read-out crossbar",
             "-" * 74,
             ""]
    for i in range(readout_power.shape[0]):
        lines.append(config.DEVICE_LABELS[i])
        lines.append("  degree %d : %s" % (degree,
                                           equation_text(readout_power[i, 0])))
        lines.append("  degree 1 : %s" % equation_text(flat_power[i, 0]))
        lines.append("")

    lines += ["-" * 74,
              "4 x %d classification crossbar" % matrix_power.shape[1],
              "-" * 74,
              ""]
    for i in range(matrix_power.shape[0]):
        lines.append(config.DEVICE_LABELS[i])
        for j in range(matrix_power.shape[1]):
            lines.append("  -> %-12s %s" % (class_names[j] + " °C",
                                            equation_text(matrix_power[i, j])))
        lines.append("")

    path = config.result_path("polynomial_weights.txt")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    print("written: %s" % path)


def main():
    config.ensure_results_dir()
    ds = data_loader.prepare(split=config.SPLIT, n_classes=config.N_CLASSES)

    print("=" * 78)
    print("POLYNOMIAL WEIGHTS")
    print("=" * 78)

    print("\ntraining the 4 x 1 read-out crossbar (degree %d and degree 1)..."
          % config.POLY_DEGREE)
    net_poly = train(ds, 1, "regression", config.POLY_DEGREE)
    net_linear = train(ds, 1, "regression", 1)
    readout_power, flat_power = plot_readout_weights(net_poly, net_linear)

    print("\nread-out synapses:")
    for i in range(readout_power.shape[0]):
        print("  %-12s %s" % (config.DEVICE_LABELS[i],
                              equation_text(readout_power[i, 0], 4)))

    print("\ntraining the 4 x %d classification crossbar..." % ds.n_classes)
    net_class = train(ds, ds.n_classes, "classification", config.POLY_DEGREE)
    matrix_power = plot_weight_matrix(net_class, ds.class_names_short)

    write_equations(readout_power, flat_power, matrix_power,
                    ds.class_names_short)
    plt.show()
    return net_poly, net_class


if __name__ == "__main__":
    readout, classifier = main()
