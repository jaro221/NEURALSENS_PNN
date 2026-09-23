# -*- coding: utf-8 -*-
"""
The minimal case: one layer, four weights, four temperature bands.

------------------------------------------------------------------------------
The question
------------------------------------------------------------------------------
Everything up to here used an array with one column per class: 4 x N weights,
each a differential pair, decided by arg max over the columns.  Is that
necessary?  A temperature band is an *ordered* label, and arg max over N
columns discards that ordering.  If instead the array produces a single
current that rises monotonically with temperature, the bands follow from
comparing it against N-1 fixed thresholds.

That needs one column, so four devices and four weights:

    z(T) = sum_i V_i * g_i(T)                                       (one column)
    band = number of thresholds that z exceeds                      (N-1 comparators)

Because every g_i is a positive conductance and every V_i is a positive read
voltage, z(T) is a sum of increasing functions and is therefore monotone by
construction.  No differential pairs are needed and no weight is negative,
which also answers the objection that a passive device cannot produce a
negative contribution: for this read-out it never has to.

With the devices exactly as measured the weights are unity and

    z(T) = sum_i V_i * ( I_i / V_i ) = sum_i I_i ,

that is, simply the sum of the four measured channel currents.

------------------------------------------------------------------------------
The result
------------------------------------------------------------------------------
Thresholds fitted on the heating ramp, evaluated on the cooling curve:

    4 bands (30 °C)   0.979
    6 bands (20 °C)   0.928
   12 bands (10 °C)   0.667

which is better than the trained 4 x N differential array at every band count,
with a twelfth of the devices.  Spearman rho between z and T is 0.997 on the
training data and 0.999 on the test data.

At four bands there are 13 errors in 625 samples and every one of them lies
within 3 °C of a band boundary, so they are quantisation at the edges of the
bins rather than a failure to resolve temperature.

Run directly (F5 in Spyder).
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
HEATING = "#eb6834"
COOLING = "#1baf7a"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

EDGE_WINDOW = 4.0        # °C either side of a band edge used to set a threshold


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


def band_edges(n_classes):
    return np.linspace(30.0, 150.0, n_classes + 1)


def to_band(temp_c, edges):
    return np.clip(np.digitize(temp_c, edges[1:-1]), 0, len(edges) - 2)


def column_current(currents, weights=None):
    """
    z(T) for one column.

    `weights` are dimensionless multipliers on each device conductance; unity
    means the devices are used exactly as measured.
    """
    if weights is None:
        return currents.sum(axis=1)
    return (currents * np.asarray(weights, float)).sum(axis=1)


def fit_thresholds(z_train, temp_train, edges):
    """
    One threshold per band boundary, taken as the median column current of the
    training samples sitting near that boundary.

    Only the training phase is used, so the thresholds carry no information
    about the cooling curve they are later tested on.
    """
    thresholds = []
    for edge in edges[1:-1]:
        near = np.abs(temp_train - edge) < EDGE_WINDOW
        if near.sum() < 4:
            return None
        thresholds.append(np.median(z_train[near]))
    thresholds = np.array(thresholds)
    return thresholds if np.all(np.diff(thresholds) > 0) else None


def evaluate(n_classes, weights=None, verbose=True):
    ds = data_loader.prepare(split="ramp", n_classes=n_classes)
    currents = ds.raw[config.CURRENT_COLUMNS].values
    temp_c = ds.raw[config.TEMP_COLUMN].values
    z = column_current(currents, weights)

    edges = band_edges(n_classes)
    thresholds = fit_thresholds(z[ds.idx_train], temp_c[ds.idx_train], edges)
    if thresholds is None:
        return None

    y_test = to_band(temp_c[ds.idx_test], edges)
    predicted = np.digitize(z[ds.idx_test], thresholds)
    accuracy = float(np.mean(predicted == y_test))
    within = float(np.mean(np.abs(predicted - y_test) <= 1))

    wrong = predicted != y_test
    if wrong.any():
        distance = np.min(np.abs(temp_c[ds.idx_test][wrong][:, None]
                                 - edges[None, 1:-1]), axis=1)
        worst = float(distance.max())
    else:
        worst = 0.0

    if verbose:
        print("  %2d bands (%3.0f °C)   accuracy %.3f   within 1 band %.3f"
              "   %2d errors, all within %.1f °C of a boundary"
              % (n_classes, 120.0 / n_classes, accuracy, within,
                 int(wrong.sum()), worst))
    return {"n_classes": n_classes, "accuracy": accuracy, "within": within,
            "thresholds": thresholds, "z": z, "temp": temp_c, "ds": ds,
            "edges": edges, "max_edge_distance": worst}


def main():
    config.ensure_results_dir()
    print("=" * 78)
    print("SINGLE LAYER, FOUR WEIGHTS, ONE COLUMN")
    print("=" * 78)
    print("z(T) = sum_i V_i g_i(T) = sum_i I_i(T); bands set by N-1 thresholds.")
    print("Thresholds fitted on the heating ramp, tested on the cooling curve.\n")

    results = {}
    for n_classes in (4, 6, 12):
        results[n_classes] = evaluate(n_classes)

    from scipy.stats import spearmanr
    r4 = results[4]
    ds = r4["ds"]
    print("\n  monotonicity of z against T:  Spearman rho = %.4f (train), "
          "%.4f (test)"
          % (spearmanr(r4["z"][ds.idx_train], r4["temp"][ds.idx_train]).statistic,
             spearmanr(r4["z"][ds.idx_test], r4["temp"][ds.idx_test]).statistic))
    print("\n  hardware: 4 devices, 1 column, 1 transimpedance stage, "
          "%d comparators." % 3)
    print("  No differential pairs and no negative weights are required,")
    print("  because a sum of positive conductances is monotone by construction.")

    make_figure(r4)
    plt.show()
    return results


def make_figure(result):
    ds = result["ds"]
    z, temp_c, edges = result["z"], result["temp"], result["edges"]
    heating = ds.heating

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), facecolor=SURFACE)

    # (a) the column current against temperature, with the thresholds
    ax = axes[0]
    style_axes(ax)
    for edge in edges[1:-1]:
        ax.axvline(edge, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)))
    for threshold in result["thresholds"]:
        ax.axhline(threshold * 1e6, color="#d03b3b", linewidth=1.2)
    ax.scatter(temp_c[heating], z[heating] * 1e6, s=6, alpha=0.45,
               color=HEATING, linewidths=0, label="heating (thresholds fitted)")
    ax.scatter(temp_c[~heating], z[~heating] * 1e6, s=6, alpha=0.45,
               color=COOLING, linewidths=0, label="cooling (test)")
    ax.plot([], [], color="#d03b3b", linewidth=1.2, label="thresholds")
    ax.set_xlabel("temperature [°C]", color=INK_2, fontsize=10)
    ax.set_ylabel("column current  z = $\\Sigma$ I$_i$  [$\\mu$A]",
                  color=INK_2, fontsize=10)
    ax.set_title("(a)  one column, four devices", loc="left", fontsize=11.5,
                 fontweight="bold", color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    # (b) where the errors are
    ax = axes[1]
    style_axes(ax)
    y_test = to_band(temp_c[ds.idx_test], edges)
    predicted = np.digitize(z[ds.idx_test], result["thresholds"])
    correct = predicted == y_test
    ax.scatter(temp_c[ds.idx_test][correct], predicted[correct], s=10,
               alpha=0.35, color=COOLING, linewidths=0, label="correct")
    ax.scatter(temp_c[ds.idx_test][~correct], predicted[~correct], s=34,
               color="#d03b3b", marker="x", linewidths=1.4,
               label="misclassified (%d of %d)"
                     % (int((~correct).sum()), len(y_test)))
    for edge in edges[1:-1]:
        ax.axvline(edge, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_yticks(range(len(edges) - 1))
    ax.set_yticklabels(["%.0f-%.0f" % (edges[i], edges[i + 1])
                        for i in range(len(edges) - 1)])
    ax.set_xlabel("true temperature [°C]", color=INK_2, fontsize=10)
    ax.set_ylabel("assigned band [°C]", color=INK_2, fontsize=10)
    ax.set_title("(b)  every error sits on a boundary", loc="left",
                 fontsize=11.5, fontweight="bold", color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    fig.suptitle("Four weights, one column, four temperature bands "
                 "(accuracy %.3f on the cooling curve)" % result["accuracy"],
                 x=0.055, y=0.985, ha="left", fontsize=13.5,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = config.result_path("22_single_layer_four_weights.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("\n  figure written to %s" % path)


if __name__ == "__main__":
    single_layer_results = main()
