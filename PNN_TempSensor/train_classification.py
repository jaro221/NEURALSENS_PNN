# -*- coding: utf-8 -*-
"""
Step 3 - classify the temperature into bands (in-sensor decision making).

Regression gives a number; a sensor that has to raise an alarm needs a
decision.  This script sorts every reading into one of N_CLASSES equal-width
temperature bands directly from the four currents, which is the in-sensor
computing task the project is aiming at: the answer is produced where the
measurement happens, and only the class index has to leave the chip.

The same four models as in train_regression.py are compared:

    single device       threshold the best single-device calibration
    linear crossbar     4 x N crossbar, constant synapses      (degree 1)
    PNN crossbar        4 x N crossbar, polynomial synapses    (degree POLY_DEGREE)
    conventional ANN    linear synapses plus a tanh hidden layer

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
N_CLASSES = config.N_CLASSES
EPOCHS = config.EPOCHS
DEGREE_SCAN = True
SCAN_DEGREES = (1, 2, 3, 4, 5)
SCAN_SEEDS = (0, 1, 2)
# ---------------------------------------------------------------------------


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def per_class_report(y_true, y_pred, n_classes):
    """Precision, recall and F1 per class, computed without sklearn."""
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    hit = np.diag(matrix).astype(float)

    precision = np.divide(hit, predicted, out=np.zeros(n_classes),
                          where=predicted > 0)
    recall = np.divide(hit, support, out=np.zeros(n_classes), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros(n_classes), where=denominator > 0)
    return precision, recall, f1, support


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def within_one_class(y_true, y_pred):
    """
    Fraction predicted into the correct band or a neighbouring one.

    The bands are an arbitrary grid laid over a continuous quantity, so a
    reading that lands one band off is a small error, not a wrong answer.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def train_pnn(ds, degree, hidden=(), seed=0, verbose=0):
    net = PNN(n_in=ds.X_train.shape[1], n_out=ds.n_classes, hidden=hidden,
              degree=degree, basis=config.POLY_BASIS, task="classification",
              poly_hidden=config.POLY_HIDDEN, seed=seed)
    net.fit(ds.X_train, ds.y_class_train, ds.X_val, ds.y_class_val,
            epochs=EPOCHS, batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE, l2=config.L2, clip=config.GRAD_CLIP,
            patience=config.PATIENCE, seed=seed, verbose=verbose)
    return net


def main():
    config.ensure_results_dir()
    ds = data_loader.prepare(split=SPLIT, n_classes=N_CLASSES)

    print("=" * 78)
    print("TEMPERATURE BAND CLASSIFICATION (%d classes)" % N_CLASSES)
    print("=" * 78)
    print(ds.describe())
    print("\nclasses:")
    for name in ds.class_names:
        print("   ", name)
    if SPLIT == "ramp":
        print("\n  trained on the heating ramp, tested on the cooling curve")

    results = {}
    predictions = {}
    models = {}

    # --- reference 1: threshold the single-device calibration -------------
    train_mask = np.zeros(len(ds.raw), dtype=bool)
    train_mask[ds.idx_train] = True
    calibrations = device_fit.fit_inverse_calibration(ds.raw, train_mask,
                                                      degree=DEGREE)
    best_col, best_acc = None, -1.0
    for col in config.CURRENT_COLUMNS:
        temperature = device_fit.apply_inverse_calibration(
            calibrations[col], ds.raw[col].values[ds.idx_test])
        labels, _ = data_loader.make_temperature_classes(
            temperature, ds.n_classes, ds.class_edges)
        if accuracy(ds.y_class_test, labels) > best_acc:
            best_col = col
            best_acc = accuracy(ds.y_class_test, labels)
            predictions["single device"] = labels
    results["single device"] = best_acc
    label = config.DEVICE_LABELS[config.CURRENT_COLUMNS.index(best_col)]
    print("\nbest single-device calibration: %s" % label)

    # --- the two crossbars -------------------------------------------------
    print("\nTraining linear crossbar (degree 1, no hidden layer)...")
    net_linear = train_pnn(ds, degree=1, hidden=())
    models["linear crossbar"] = net_linear
    predictions["linear crossbar"] = net_linear.predict(ds.X_test)

    print("Training polynomial crossbar (degree %d, no hidden layer)..." % DEGREE)
    net_poly = train_pnn(ds, degree=DEGREE, hidden=())
    models["PNN crossbar"] = net_poly
    predictions["PNN crossbar"] = net_poly.predict(ds.X_test)
    print(net_poly.summary())

    print("\nTraining conventional ANN (linear synapses + tanh hidden layer)...")
    net_ann = train_pnn(ds, degree=1, hidden=config.BASELINE_HIDDEN)
    models["conventional ANN"] = net_ann
    predictions["conventional ANN"] = net_ann.predict(ds.X_test)

    for name in ("linear crossbar", "PNN crossbar", "conventional ANN"):
        results[name] = accuracy(ds.y_class_test, predictions[name])

    # --- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TEST SET RESULTS (%s split, %d samples)" % (SPLIT, len(ds.T_test)))
    print("=" * 78)
    print("%-22s %10s %14s %8s" % ("model", "accuracy", "within 1 band", "params"))
    for name in predictions:
        n_par = models[name].n_parameters() if name in models else DEGREE + 1
        print("%-22s %10.4f %14.4f %8d"
              % (name, results[name],
                 within_one_class(ds.y_class_test, predictions[name]), n_par))

    print("\nPer-class results for the PNN crossbar:")
    precision, recall, f1, support = per_class_report(
        ds.y_class_test, predictions["PNN crossbar"], ds.n_classes)
    print("  %-16s %10s %8s %8s %8s"
          % ("class", "precision", "recall", "F1", "support"))
    for i, name in enumerate(ds.class_names_short):
        print("  %-16s %10.3f %8.3f %8.3f %8d"
              % (name + " °C", precision[i], recall[i], f1[i], support[i]))

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
                runs.append((accuracy(ds.y_class_train, net.predict(ds.X_train)),
                             accuracy(ds.y_class_test, net.predict(ds.X_test))))
            runs = np.array(runs)
            scan.append({"degree": degree, "train": runs[:, 0].mean(),
                         "test": runs[:, 1].mean(), "test_std": runs[:, 1].std(),
                         "n_par": net.n_parameters()})
            print("  degree %d:  train %.4f   test %.4f +- %.4f   (%2d parameters)"
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


def _draw_confusion(ax, matrix, title, names, cmap):
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ax.imshow(normalised, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted band [°C]", fontsize=9)
    ax.set_ylabel("true band [°C]", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=8,
                        color="white" if normalised[i, j] > 0.5 else "black")


def make_figures(ds, results, predictions, models, scan):
    order = list(predictions.keys())
    names = ds.class_names_short
    cmaps = {"single device": "Greys", "linear crossbar": "Blues",
             "PNN crossbar": "Reds", "conventional ANN": "Greens"}

    # 09 - confusion matrices
    fig, axes = plt.subplots(1, len(order), figsize=(4.1 * len(order), 4.3))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, order):
        matrix = confusion_matrix(ds.y_class_test, predictions[name], ds.n_classes)
        _draw_confusion(ax, matrix,
                        "%s\naccuracy = %.3f" % (name, results[name]),
                        names, cmaps[name])
    fig.suptitle("Confusion matrices on the test set (%s split)" % ds.split,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.result_path("09_classification_confusion_matrices.png"),
                dpi=config.DPI, bbox_inches="tight")

    # 10 - accuracy comparison
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    values = [results[n] for n in order]
    bars = ax.bar(order, values, color=[PALETTE[n] for n in order],
                  edgecolor="black", linewidth=1.2, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                "%.3f" % value, ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.axhline(1.0 / ds.n_classes, color="black", linestyle=":",
               label="chance level")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Temperature band classification (%s split)" % ds.split,
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.result_path("10_classification_accuracy.png"),
                dpi=config.DPI, bbox_inches="tight")

    # 11 - training history
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in ("linear crossbar", "PNN crossbar", "conventional ANN"):
        net = models[name]
        axes[0].plot(net.history["train_loss"], color=PALETTE[name], label=name)
        axes[1].plot(net.history["val_metric"], color=PALETTE[name], label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Training loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title("Validation accuracy", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.result_path("11_classification_training_history.png"),
                dpi=config.DPI, bbox_inches="tight")

    # 12 - predictions along the test sequence
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.plot(ds.T_test, ds.y_class_test, color="black", linewidth=2,
            label="true band")
    for name in ("linear crossbar", "PNN crossbar"):
        ax.scatter(ds.T_test, predictions[name], s=12, alpha=0.5,
                   color=PALETTE[name], label=name)
    ax.set_xlabel("true temperature [°C]")
    ax.set_ylabel("class index")
    ax.set_yticks(range(ds.n_classes))
    ax.set_yticklabels(names)
    ax.set_title("Predicted band against the measured temperature",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.result_path("12_classification_predictions.png"),
                dpi=config.DPI, bbox_inches="tight")

    # 13 - degree scan
    if scan:
        fig, ax = plt.subplots(figsize=config.FIGSIZE)
        degrees = [s["degree"] for s in scan]
        ax.plot(degrees, [s["train"] for s in scan], "o-", color="#2980b9",
                label="train")
        ax.errorbar(degrees, [s["test"] for s in scan],
                    yerr=[s["test_std"] for s in scan], fmt="s-",
                    color="#c0392b", capsize=4, label="test")
        ax.set_xlabel("degree of the synaptic polynomials")
        ax.set_ylabel("accuracy")
        ax.set_title("Classification accuracy against synapse degree\n"
                     "(degree 1 = constant weights = conventional crossbar)",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(degrees)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(config.result_path("13_classification_degree_scan.png"),
                    dpi=config.DPI, bbox_inches="tight")

    print("\nFigures written to %s" % config.RESULTS_DIR)


if __name__ == "__main__":
    dataset, results, predictions, models = main()
