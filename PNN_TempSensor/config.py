# -*- coding: utf-8 -*-
"""
Central configuration for the PNN (Physical Neural Network) experiments.

Everything that you may want to tune while working in Spyder lives here.
Edit a value, save, and re-run any of the train_*.py scripts.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CSV_NAME = "DATA_[1,2V,1,4V;1,6;1.8]_TempSweep_2026-02-18_19-44-46.csv"


def _this_dir():
    """Directory of this file, with a fallback for odd Spyder run modes."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:                                    # pragma: no cover
        return os.path.abspath(os.getcwd())


PACKAGE_DIR = _this_dir()
RESULTS_DIR = os.path.join(PACKAGE_DIR, "results")


def find_csv():
    """
    Locate the measurement CSV.

    Looks in this folder, the parent folder and the current working directory,
    so the scripts run whatever Spyder has set as the working directory.
    """
    candidates = [
        os.path.join(PACKAGE_DIR, CSV_NAME),
        os.path.join(os.path.dirname(PACKAGE_DIR), CSV_NAME),
        os.path.join(os.getcwd(), CSV_NAME),
        os.path.join(os.path.dirname(os.getcwd()), CSV_NAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find '%s'.\nLooked in:\n  %s" % (CSV_NAME, "\n  ".join(candidates))
    )


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def result_path(filename):
    return os.path.join(ensure_results_dir(), filename)


# ---------------------------------------------------------------------------
# Sensor / measurement description
# ---------------------------------------------------------------------------

CURRENT_COLUMNS = ["I_D1_1.2V", "I_D2_1.4V", "I_D3_1.6V", "I_D4_1.8V"]
DEVICE_LABELS = ["D1 @ 1.2 V", "D2 @ 1.4 V", "D3 @ 1.6 V", "D4 @ 1.8 V"]
BIAS_VOLTAGES = [1.2, 1.4, 1.6, 1.8]
TEMP_COLUMN = "act_temp"

# The heating ramp ends where set_temp drops from 150 back to 30; after that
# the plate cools down freely.  Detected automatically in data_loader, this is
# only the fallback.
FALLBACK_RAMP_END = 1133

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

SEED = 0

# How to split the time series.
#   "ramp"    - train on the heating ramp, test on the cooling curve (honest)
#   "blocked" - contiguous blocks of samples, 80/20 (moderately honest)
#   "random"  - i.i.d. shuffle (optimistic: neighbouring samples leak)
SPLIT = "ramp"
BLOCK_SIZE = 25          # samples per block for "blocked" and for validation
VAL_FRACTION = 0.2

# Temperature classes: equal-width bands over the measured range.
# 12 classes gives 10 °C bands, which is the step the plate was actually
# driven with.  Coarser bands (6) make every model look equally good.
N_CLASSES = 12

# Polynomial weights
POLY_DEGREE = 3          # degree of every synaptic polynomial P_ij(u)
POLY_BASIS = "chebyshev"  # "chebyshev" | "legendre" | "power"

# Network shape.
#
# HIDDEN = () is the physical layout the project proposes: one crossbar of
# polynomial devices, no hidden neurons and no activation function, so the
# device characteristic has to supply all of the nonlinearity.  This is the
# configuration in which polynomial synapses pay off most clearly.
# Give HIDDEN a width, e.g. (16,), to put a conventional hidden layer behind
# the crossbar instead.
HIDDEN = ()
POLY_HIDDEN = False      # True -> hidden layers also use polynomial weights
BASELINE_HIDDEN = (16,)  # hidden layer used for the conventional-ANN baseline

# Optimisation
EPOCHS = 600
BATCH_SIZE = 32
LEARNING_RATE = 0.02
L2 = 1e-5
GRAD_CLIP = 5.0
PATIENCE = 80            # early stopping patience (epochs)

# Figures
DPI = 200
FIGSIZE = (11, 6)
