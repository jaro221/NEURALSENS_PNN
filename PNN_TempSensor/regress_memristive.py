# -*- coding: utf-8 -*-
"""
Continuous temperature read-out (regression) from the memristive sensor.

------------------------------------------------------------------------------
Is regression the right formulation?
------------------------------------------------------------------------------
Yes, and it is better than the band classification of single_layer.py.
Temperature is continuous and ordered, the band grid is arbitrary, and
classifying quantises the answer before it leaves the chip: the reported
accuracy then depends on how wide the bands were chosen to be, while the mean
absolute error does not.  Everything below is therefore stated as MAE in °C.

------------------------------------------------------------------------------
Where the nonlinearity has to live
------------------------------------------------------------------------------
This file also records a negative result, because it changes the architecture.

The obvious plan is to train the programmed states (C_pf, Ea) of a crossbar so
that its column currents, passed through a linear read-out, give temperature.
That does not work, and not because of tuning.  Every conductance is positive
and rises with temperature, so the column current

    z(T) = sum_i V_i * w_ij(T)

is monotone in T whatever states the devices are in.  What the read-out has to
do is *invert* that monotone curve, and inverting it is a nonlinear operation
that no choice of conductances can perform.  Training the states and a linear
read-out jointly is then a pure scale degeneracy: the conductances shrink to
their lower bound while the read-out weight grows to compensate, the prediction
stays constant, and the error sits at the standard deviation of the target
(36 °C here, i.e. predicting the mean).  Observed directly - MSE pinned at
exactly 1.000 on standardised targets while the read-out weight drifted upward.

The conclusion is structural.  For a continuous read-out the devices supply the
*signal* and the nonlinearity belongs in the read-out, not in the array.  That
read-out is peripheral electronics, where a negative coefficient costs nothing,
so the objection that a passive device cannot produce a negative contribution
does not apply to it.

------------------------------------------------------------------------------
What works
------------------------------------------------------------------------------
A polynomial read-out fitted by least squares on the heating ramp and evaluated
on the cooling curve.  No iterative training is involved at all.

Run directly (F5 in Spyder).
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]


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


def polynomial_features(X, degree, train_index):
    """
    Scale each column to [-1, 1] using the training range, then stack powers.

    The scaling makes the least-squares problem well conditioned; the clip
    keeps the polynomial from being evaluated far outside the range it was
    fitted on, which is where a cubic diverges fastest.
    """
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    lo, hi = X[train_index].min(axis=0), X[train_index].max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    u = np.clip(2.0 * (X - lo) / span - 1.0, -1.25, 1.25)
    return np.hstack([u ** k for k in range(1, degree + 1)]
                     + [np.ones((len(u), 1))])


def fit_readout(features, target, train_index):
    coefficients, *_ = np.linalg.lstsq(features[train_index], target[train_index],
                                       rcond=None)
    return coefficients


def metrics(true, pred):
    error = pred - true
    return {"MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "max": float(np.max(np.abs(error))),
            "R2": float(1 - np.sum(error ** 2)
                        / np.sum((true - true.mean()) ** 2))}


def evaluate(signal, degree, ds, temp_c):
    features = polynomial_features(signal, degree, ds.idx_train)
    coefficients = fit_readout(features, temp_c, ds.idx_train)
    prediction = features.dot(coefficients)
    return metrics(temp_c[ds.idx_test], prediction[ds.idx_test]), prediction


def main():
    config.ensure_results_dir()
    ds = data_loader.prepare(split="ramp")
    currents = ds.raw[config.CURRENT_COLUMNS].values
    temp_c = ds.raw[config.TEMP_COLUMN].values

    print("=" * 78)
    print("TEMPERATURE READ-OUT AS REGRESSION")
    print("=" * 78)
    print("Read-out fitted on the heating ramp, evaluated on the cooling")
    print("curve (%d samples).  No iterative training.\n" % len(ds.T_test))

    print("  %-42s %6s %6s %6s %8s"
          % ("configuration", "MAE", "RMSE", "max", "R2"))
    configurations = [
        ("one column, z = sum I, linear read-out",
         currents.sum(axis=1, keepdims=True), 1),
        ("one column, z = sum I, cubic read-out",
         currents.sum(axis=1, keepdims=True), 3),
        ("single channel 1.6 V, cubic read-out", currents[:, [2]], 3),
        ("four channels kept separate, linear read-out", currents, 1),
        ("four channels kept separate, quadratic read-out", currents, 2),
        ("four channels kept separate, cubic read-out", currents, 3),
    ]

    results = {}
    for name, signal, degree in configurations:
        score, prediction = evaluate(signal, degree, ds, temp_c)
        results[name] = (score, prediction)
        print("  %-42s %6.2f %6.2f %6.2f %8.4f"
              % (name, score["MAE"], score["RMSE"], score["max"], score["R2"]))

    print("\n  The read-out degree is what matters, not the number of columns:")
    print("  a linear read-out cannot invert a monotone exponential and loses")
    print("  an order of magnitude.  Keeping the four channels separate does")
    print("  not improve the average error much, but it halves the worst case,")
    print("  because the channels have different activation energies and their")
    print("  ratios carry information that summing them destroys.")

    make_figure(ds, temp_c, results)
    plt.show()
    return results


def make_figure(ds, temp_c, results):
    order = ["one column, z = sum I, linear read-out",
             "one column, z = sum I, cubic read-out",
             "four channels kept separate, cubic read-out"]
    colours = ["#7f8c8d", "#2980b9", "#c0392b"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE)
    lims = [temp_c[ds.idx_test].min() - 5, temp_c[ds.idx_test].max() + 5]

    for ax, name, colour in zip(axes[:2], order[1:], colours[1:]):
        style_axes(ax)
        score, prediction = results[name]
        ax.scatter(temp_c[ds.idx_test], prediction[ds.idx_test], s=7,
                   alpha=0.45, color=colour, linewidths=0)
        ax.plot(lims, lims, "--", color=AXIS, linewidth=1.2)
        ax.set_xlabel("true temperature [°C]", color=INK_2, fontsize=10)
        ax.set_ylabel("predicted temperature [°C]", color=INK_2, fontsize=10)
        ax.set_title("%s\nMAE %.2f °C, worst %.1f °C"
                     % (name.replace(", ", "\n"), score["MAE"], score["max"]),
                     loc="left", fontsize=9.5, fontweight="bold", color=INK,
                     pad=8)

    ax = axes[2]
    style_axes(ax)
    for name, colour in zip(order, colours):
        score, prediction = results[name]
        ax.scatter(temp_c[ds.idx_test],
                   prediction[ds.idx_test] - temp_c[ds.idx_test],
                   s=6, alpha=0.4, color=colour, linewidths=0,
                   label="%s (%.2f)" % (name.split(",")[-1].strip(),
                                        score["MAE"]))
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xlabel("true temperature [°C]", color=INK_2, fontsize=10)
    ax.set_ylabel("error [°C]", color=INK_2, fontsize=10)
    ax.set_title("error against temperature", loc="left", fontsize=9.5,
                 fontweight="bold", color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)

    fig.suptitle("Temperature as a continuous read-out, tested on the cooling curve",
                 x=0.05, y=0.99, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = config.result_path("23_regression_readout.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("\n  figure written to %s" % path)


if __name__ == "__main__":
    regression_results = main()
