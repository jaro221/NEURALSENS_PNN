# -*- coding: utf-8 -*-
"""
Step 2 - read the temperature out of the four sensor currents (regression).

The central comparison is between two crossbars of exactly the same size,
4 inputs by 1 output, with no hidden neurons and no activation function:

    linear crossbar       every synapse is a constant       (degree 1)
    polynomial crossbar   every synapse is a polynomial     (degree POLY_DEGREE)

That is the question the project asks - does giving the device a polynomial
characteristic remove the need for hidden layers - and it is asked in the
layout that would actually be fabricated.

Two further references are included:

    single-device calibration   T = P(I_k), the classical approach: one device,
                                one polynomial, fitted in device_fit.py
    conventional ANN            linear synapses plus a tanh hidden layer, i.e.
                                what you would run in software today

Run directly (F5 in Spyder).  Settings are at the top of the file.
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import data_loader
import device_fit
from pnn import PNN

# --- settings --------------------------------------------------------------
SPLIT = config.SPLIT
DEGREE = config.POLY_DEGREE
EPOCHS = config.EPOCHS
DEGREE_SCAN = True               # train degrees 1..5 and plot the trend
SCAN_DEGREES = (1, 2, 3, 4, 5)
SCAN_SEEDS = (0, 1, 2)           # averaged, so the trend is not one lucky run
# ---------------------------------------------------------------------------


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    error = y_pred - y_true
    total = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(1.0 - np.sum(error ** 2) / total) if total > 0 else np.nan,
        "max_err": float(np.max(np.abs(error))),
    }


def train_pnn(ds, degree, hidden=(), seed=0, verbose=0):
    net = PNN(n_in=ds.X_train.shape[1], n_out=1, hidden=hidden, degree=degree,
              basis=config.POLY_BASIS, task="regression",
              poly_hidden=config.POLY_HIDDEN, seed=seed)
    net.fit(ds.X_train, ds.y_reg_train, ds.X_val, ds.y_reg_val,
            epochs=EPOCHS, batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE, l2=config.L2, clip=config.GRAD_CLIP,
            patience=config.PATIENCE, seed=seed, verbose=verbose)
    return net


def predict_temperature(net, ds, X):
    """The network output is standardised; convert it back to °C."""
    return ds.target_scaler.inverse_transform(net.predict(X).ravel())


def main():
    config.ensure_results_dir()
    ds = data_loader.prepare(split=SPLIT)

    print("=" * 78)
    print("TEMPERATURE READ-OUT FROM FOUR SENSOR CURRENTS")
    print("=" * 78)
    print(ds.describe())
    if SPLIT == "ramp":
        print("\n  trained on the heating ramp, tested on the cooling curve")

    results = {}
    predictions = {}
    models = {}

    # --- reference 1: single-device polynomial calibration ----------------
    train_mask = np.zeros(len(ds.raw), dtype=bool)
    train_mask[ds.idx_train] = True
    calibrations = device_fit.fit_inverse_calibration(ds.raw, train_mask, degree=DEGREE)

    best_col, best_mae = None, np.inf
    for col in config.CURRENT_COLUMNS:
        pred = device_fit.apply_inverse_calibration(
            calibrations[col], ds.raw[col].values[ds.idx_test])
        m = metrics(ds.T_test, pred)
        if m["MAE"] < best_mae:
            best_col, best_mae = col, m["MAE"]
            predictions["single device"] = pred
            results["single device"] = m
    label = config.DEVICE_LABELS[config.CURRENT_COLUMNS.index(best_col)]
    print("\nbest single-device calibration: %s, degree %d" % (label, DEGREE))

    # --- the two crossbars -------------------------------------------------
    print("\nTraining linear crossbar (degree 1, no hidden layer)...")
    net_linear = train_pnn(ds, degree=1, hidden=())
    models["linear crossbar"] = net_linear
    predictions["linear crossbar"] = predict_temperature(net_linear, ds, ds.X_test)
    results["linear crossbar"] = metrics(ds.T_test, predictions["linear crossbar"])

    print("Training polynomial crossbar (degree %d, no hidden layer)..." % DEGREE)
    net_poly = train_pnn(ds, degree=DEGREE, hidden=())
    models["PNN crossbar"] = net_poly
    predictions["PNN crossbar"] = predict_temperature(net_poly, ds, ds.X_test)
    results["PNN crossbar"] = metrics(ds.T_test, predictions["PNN crossbar"])
    print(net_poly.summary())

    # --- reference 2: conventional ANN ------------------------------------
    print("\nTraining conventional ANN (linear synapses + tanh hidden layer)...")
    net_ann = train_pnn(ds, degree=1, hidden=config.BASELINE_HIDDEN)
    models["conventional ANN"] = net_ann
    predictions["conventional ANN"] = predict_temperature(net_ann, ds, ds.X_test)
    results["conventional ANN"] = metrics(ds.T_test, predictions["conventional ANN"])

    # --- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TEST SET RESULTS (%s split, %d samples)" % (SPLIT, len(ds.T_test)))
    print("=" * 78)
    print("%-22s %10s %8s %8s %9s %7s"
          % ("model", "MAE[°C]", "RMSE", "R2", "max err", "params"))
    for name, m in results.items():
        n_par = models[name].n_parameters() if name in models else DEGREE + 1
        print("%-22s %10.2f %8.2f %8.4f %9.2f %7d"
              % (name, m["MAE"], m["RMSE"], m["R2"], m["max_err"], n_par))

    gain = results["linear crossbar"]["MAE"] / results["PNN crossbar"]["MAE"]
    print("\npolynomial synapses reduce the crossbar error by a factor of %.1f"
          % gain)

    # --- degree scan -------------------------------------------------------
    scan = None
    if DEGREE_SCAN:
        print("\nScanning the polynomial degree of the crossbar "
              "(%d seeds each)..." % len(SCAN_SEEDS))
        scan = []
        for degree in SCAN_DEGREES:
            runs = []
            for seed in SCAN_SEEDS:
                net = train_pnn(ds, degree=degree, hidden=(), seed=seed)
                runs.append((
                    metrics(ds.T_train, predict_temperature(net, ds, ds.X_train))["MAE"],
                    metrics(ds.T_test, predict_temperature(net, ds, ds.X_test))["MAE"],
                ))
            runs = np.array(runs)
            scan.append({"degree": degree,
                         "train": runs[:, 0].mean(),
                         "test": runs[:, 1].mean(),
                         "test_std": runs[:, 1].std(),
                         "n_par": net.n_parameters()})
            print("  degree %d:  train MAE %6.2f   test MAE %6.2f +- %.2f   "
                  "(%2d parameters)"
                  % (degree, scan[-1]["train"], scan[-1]["test"],
                     scan[-1]["test_std"], scan[-1]["n_par"]))

    make_figures(ds, results, predictions, models, scan)
    plt.show()
    return ds, results, predictions, models


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

PALETTE = {
    "single device": "#7f8c8d",
    "linear crossbar": "#2980b9",
    "PNN crossbar": "#c0392b",
    "conventional ANN": "#27ae60",
}


def make_figures(ds, results, predictions, models, scan):
    order = list(predictions.keys())
    colors = [PALETTE[n] for n in order]

    # 04 - predicted vs true
    fig, axes = plt.subplots(1, len(order), figsize=(4.0 * len(order), 4.3),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = ds.T_test.min() - 5, ds.T_test.max() + 5
    for ax, name, color in zip(axes, order, colors):
        ax.scatter(ds.T_test, predictions[name], s=9, alpha=0.5, color=color)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title("%s\nMAE = %.2f °C" % (name, results[name]["MAE"]),
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("true temperature [°C]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("predicted temperature [°C]")
    fig.suptitle("Temperature read-out on the test set (%s split)" % ds.split,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.result_path("04_regression_predicted_vs_true.png"),
                dpi=config.DPI, bbox_inches="tight")

    # 05 - error against temperature
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    for name, color in zip(order, colors):
        ax.scatter(ds.T_test, predictions[name] - ds.T_test, s=9, alpha=0.5,
                   color=color, label=name)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("true temperature [°C]")
    ax.set_ylabel("prediction error [°C]")
    ax.set_title("Read-out error across the measured range",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.result_path("05_regression_error_vs_temperature.png"),
                dpi=config.DPI, bbox_inches="tight")

    # 06 - training history of the two crossbars
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    for name in ("linear crossbar", "PNN crossbar"):
        net = models[name]
        ax.plot(net.history["train_loss"], color=PALETTE[name],
                label="%s - train" % name)
        ax.plot(net.history["val_loss"], color=PALETTE[name], linestyle="--",
                label="%s - validation" % name)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised target)")
    ax.set_title("Training history of the two crossbars",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.result_path("06_regression_training_history.png"),
                dpi=config.DPI, bbox_inches="tight")

    # 07 - MAE comparison
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    names = list(results.keys())
    values = [results[n]["MAE"] for n in names]
    bars = ax.bar(names, values, color=[PALETTE[n] for n in names],
                  edgecolor="black", linewidth=1.2, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                "%.2f" % value, ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("test MAE [°C]")
    ax.set_title("Temperature read-out accuracy (%s split)" % ds.split,
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(config.result_path("07_regression_model_comparison.png"),
                dpi=config.DPI, bbox_inches="tight")

    # 08 - degree scan
    if scan:
        fig, ax = plt.subplots(figsize=config.FIGSIZE)
        degrees = [s["degree"] for s in scan]
        ax.plot(degrees, [s["train"] for s in scan], "o-",
                color="#2980b9", label="train")
        ax.errorbar(degrees, [s["test"] for s in scan],
                    yerr=[s["test_std"] for s in scan], fmt="s-",
                    color="#c0392b", capsize=4, label="test")
        ax.set_xlabel("degree of the synaptic polynomials")
        ax.set_ylabel("MAE [°C]")
        ax.set_title("Crossbar accuracy against synapse degree\n"
                     "(degree 1 = constant weights = conventional crossbar)",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(degrees)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(config.result_path("08_regression_degree_scan.png"),
                    dpi=config.DPI, bbox_inches="tight")

    print("\nFigures written to %s" % config.RESULTS_DIR)


if __name__ == "__main__":
    dataset, results, predictions, models = main()
