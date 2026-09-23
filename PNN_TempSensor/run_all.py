# -*- coding: utf-8 -*-
"""
Run the whole study in order and write every figure into results/.

    0. plot_data              one overview image of the raw measurement
    1. device_fit             characterise the measured devices
    2. train_regression       read the temperature out of the four currents
    3. train_classification   sort readings into temperature bands
    4. weight_analysis        what the trained synapses cost in hardware
    5. plot_weights           the polynomial weights themselves

In Spyder this takes a few minutes.  To look at one part only, open the
corresponding file and run that instead - each script is self-contained.

The split comparison at the end is worth reading: it shows how much the
reported accuracy depends on how the sweep is divided, and in particular what
it costs to ask the model for a temperature while the plate is cooling after
it was only ever shown the plate heating up.
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import config
import data_loader
from pnn import PNN


def _run(name, function):
    print("\n\n" + "#" * 78)
    print("# " + name)
    print("#" * 78)
    start = time.time()
    result = function()
    print("\n[%s finished in %.1f s]" % (name, time.time() - start))
    return result


def split_comparison():
    """
    Train the same crossbar under all three split modes.

    "random" and "blocked" both test on the heating ramp, the regime the model
    was fitted on, and land in the same place.  "ramp" is the one that costs
    something, because its test set is the cooling curve: the devices show real
    thermal hysteresis, so this asks the model to work in a regime it never
    saw.  That gap is a property of the sensor, not of the fitting procedure,
    and it is the number that describes what the device would do in the field -
    which is why "ramp" is the default in config.py.
    """
    print("\n\n" + "#" * 78)
    print("# How much the split matters")
    print("#" * 78)
    print("\nSame crossbar (degree %d), three ways of dividing the same sweep:\n"
          % config.POLY_DEGREE)
    print("  %-10s %12s %12s   %s"
          % ("split", "train MAE", "test MAE", "what the test set is"))

    explanation = {
        "random": "shuffled samples, same thermal regime",
        "blocked": "unseen stretches, same thermal regime",
        "ramp": "the cooling curve, a regime never trained on",
    }

    rows = []
    for mode in ("random", "blocked", "ramp"):
        ds = data_loader.prepare(split=mode)
        net = PNN(n_in=ds.X_train.shape[1], n_out=1, hidden=config.HIDDEN,
                  degree=config.POLY_DEGREE, basis=config.POLY_BASIS,
                  task="regression", seed=config.SEED)
        net.fit(ds.X_train, ds.y_reg_train, ds.X_val, ds.y_reg_val,
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                lr=config.LEARNING_RATE, l2=config.L2, clip=config.GRAD_CLIP,
                patience=config.PATIENCE, seed=config.SEED, verbose=0)

        train_mae = np.mean(np.abs(
            ds.target_scaler.inverse_transform(net.predict(ds.X_train).ravel())
            - ds.T_train))
        test_mae = np.mean(np.abs(
            ds.target_scaler.inverse_transform(net.predict(ds.X_test).ravel())
            - ds.T_test))
        rows.append((mode, train_mae, test_mae))
        print("  %-10s %9.2f °C %9.2f °C   %s"
              % (mode, train_mae, test_mae, explanation[mode]))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    modes = [r[0] for r in rows]
    ax.bar(modes, [r[2] for r in rows],
           color=["#95a5a6", "#2980b9", "#c0392b"],
           edgecolor="black", linewidth=1.2, width=0.55)
    for i, row in enumerate(rows):
        ax.text(i, row[2] + 0.05, "%.2f" % row[2], ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("test MAE [°C]")
    ax.set_title("The same model under three different splits",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(config.result_path("17_split_comparison.png"),
                dpi=config.DPI, bbox_inches="tight")
    return rows


def main():
    config.ensure_results_dir()
    print("data file : %s" % config.find_csv())
    print("results   : %s" % config.RESULTS_DIR)
    print("split     : %s" % config.SPLIT)
    print("degree    : %d (%s basis)" % (config.POLY_DEGREE, config.POLY_BASIS))

    import plot_data
    import device_fit
    import train_regression
    import train_classification
    import weight_analysis
    import plot_weights

    _run("0. the measurement itself", plot_data.main)
    _run("1. device characterisation", device_fit.main)
    _run("2. temperature read-out (regression)", train_regression.main)
    _run("3. temperature bands (classification)", train_classification.main)
    _run("4. synapses as hardware", weight_analysis.main)
    _run("5. the polynomial weights", plot_weights.main)
    split_comparison()

    print("\n\nAll done.  Figures are in %s" % config.RESULTS_DIR)
    plt.show()


if __name__ == "__main__":
    # Keep the figures from piling up on screen during a full run.
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.close("all")
    main()
