# -*- coding: utf-8 -*-
"""
Loading, cleaning, labelling and splitting of the temperature-sweep
measurement.

The measurement
---------------
1802 samples taken every 2 s while four diodes are read at four bias points
(1.2 / 1.4 / 1.6 / 1.8 V).  The plate is driven from 30 to 150 °C in 10 °C
steps (samples 0..~1132) and then left to cool freely back to ~31 °C
(samples ~1133..1801).

Why the split matters
---------------------
Consecutive samples are 2 s apart and are almost identical, so an i.i.d.
shuffle puts near-duplicates of every test sample into the training set and
reports an accuracy that the device would never reach in the field.  The
measurement also shows real thermal hysteresis - at 60 °C D1 delivers ~214 nA
while heating but ~157 nA while cooling - so heating and cooling are genuinely
different operating regimes.

Three split modes are therefore provided:

    "ramp"     train on the heating ramp, test on the cooling curve.
               The honest generalisation test, and the default.
    "blocked"  contiguous blocks of samples dealt out to train/val/test.
    "random"   i.i.d. shuffle.  Optimistic; kept for comparison with the
               earlier scripts in the parent folder.

Feature scaling is fitted on the training part only.
"""

import numpy as np
import pandas as pd

import config


class Dataset(object):
    """Container for one prepared experiment."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def describe(self):
        lines = [
            "Dataset  split=%s  features=%d" % (self.split, self.X_train.shape[1]),
            "  train %5d samples   %.1f - %.1f °C" % (
                len(self.idx_train), self.T_train.min(), self.T_train.max()),
            "  val   %5d samples   %.1f - %.1f °C" % (
                len(self.idx_val), self.T_val.min(), self.T_val.max()),
            "  test  %5d samples   %.1f - %.1f °C" % (
                len(self.idx_test), self.T_test.min(), self.T_test.max()),
        ]
        if self.n_classes:
            counts = np.bincount(self.y_class_train, minlength=self.n_classes)
            lines.append("  train class counts: %s" % counts.tolist())
            counts = np.bincount(self.y_class_test, minlength=self.n_classes)
            lines.append("  test  class counts: %s" % counts.tolist())
        if getattr(self, "n_dropouts", 0):
            temps = self.dropout_temperatures
            lines.insert(1, "  %d read-out dropouts removed (%.0f-%.0f °C)"
                         % (self.n_dropouts, temps.min(), temps.max()))
        if self.clipped_fraction > 0:
            lines.append("  %.2f %% of scaled feature values were outside the "
                         "training range and were clipped"
                         % (100 * self.clipped_fraction))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------

def find_dropouts(data, window=101, factor=0.5, n_sigma=6.0):
    """
    Flag read-out dropouts.

    The sweep contains bursts in which the current collapses by one to two
    orders of magnitude on one or more channels and then recovers, while the
    temperature does not move.  Two examples:

        t = 1906..1955 s, 130-139 °C : D4 falls from ~12 uA to ~150 nA
        t = 3400..3452 s,  33-35 °C  : all four channels fall to ~0 nA,
                                         with both neighbours reading normally
                                         (D4 = 2075 nA on either side)

    Temperature does not jump at those moments, so these are instrument faults
    (range change / contact), not device physics.  In total 65 samples, 3.6 %.

    Leaving them in is actively harmful.  They are the lowest currents in the
    run, so they anchor the bottom of the feature scaling and teach the network
    that a near-zero current means whatever temperature happened to be on the
    plate at the time.  A polynomial synapse has more freedom to fit that than
    a constant weight does, so the PNN is hurt by it more than a linear model.

    A sample is flagged when, on any channel, it is
      * below `factor` times the rolling median of its neighbourhood, and
      * below it by more than `n_sigma` times the channel read-out noise.

    The second condition keeps the test from firing on ordinary scatter, where
    a sample sits below a small median by an amount that is merely noise.

    `window` must be comfortably more than twice the longest burst, otherwise
    the burst supplies its own median and hides itself: the 27-sample burst at
    33 °C is invisible at window=51 and is caught from window=101 upwards.
    The flagged set is identical at 101, 151 and 201, so the choice is not
    delicate - it just has to be big enough.

    Returns a boolean array, True for samples to discard.
    """
    currents = data[config.CURRENT_COLUMNS]

    # Robust read-out noise from consecutive differences.
    diffs = np.diff(currents.values, axis=0)
    mad = np.median(np.abs(diffs - np.median(diffs, axis=0)), axis=0)
    sigma = 1.4826 * mad / np.sqrt(2.0)
    sigma[sigma == 0] = np.finfo(float).eps

    median = currents.rolling(window, center=True, min_periods=5).median().values
    shortfall = median - currents.values

    flagged = (currents.values < factor * median) & (shortfall > n_sigma * sigma)
    return flagged.any(axis=1)


def load_raw(path=None, drop_settling=True, remove_dropouts=True, report=False,
             return_info=False):
    """
    Read the CSV and return the cleaned DataFrame.

    The very first row reports act_temp = 40 °C while set_temp is 30 °C -
    a start-up artefact of the temperature readout - and is dropped.
    Read-out dropouts are removed as well; see find_dropouts.
    """
    path = config.find_csv() if path is None else path
    data = pd.read_csv(path)

    needed = [config.TEMP_COLUMN, "set_temp", "time"] + config.CURRENT_COLUMNS
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError("CSV is missing column(s): %s" % missing)

    if drop_settling:
        data = data.iloc[1:].reset_index(drop=True)

    data = data.dropna(subset=needed).reset_index(drop=True)

    info = {"n_dropouts": 0, "dropout_temperatures": np.array([])}
    if remove_dropouts:
        bad = find_dropouts(data)
        info["n_dropouts"] = int(bad.sum())
        info["dropout_temperatures"] = data[config.TEMP_COLUMN].values[bad]
        if report and bad.any():
            temps = info["dropout_temperatures"]
            print("removed %d read-out dropouts (%.1f %% of samples), "
                  "at %.0f-%.0f °C" % (bad.sum(), 100.0 * bad.mean(),
                                         temps.min(), temps.max()))
        data = data.loc[~bad].reset_index(drop=True)

    return (data, info) if return_info else data


def find_ramp_end(data):
    """
    Index of the first sample after the heating ramp, i.e. where set_temp is
    commanded back down.  Falls back to config.FALLBACK_RAMP_END.
    """
    set_temp = data["set_temp"].values
    drops = np.where(np.diff(set_temp) < 0)[0]
    if drops.size:
        return int(drops[0] + 1)
    return min(config.FALLBACK_RAMP_END, len(data))


def phase_mask(data):
    """Boolean array, True for samples on the heating ramp."""
    heating = np.zeros(len(data), dtype=bool)
    heating[:find_ramp_end(data)] = True
    return heating


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def make_temperature_classes(temperatures, n_classes, edges=None):
    """
    Equal-width temperature bands over the measured range.

    Returns (labels, edges).  Labels are integers 0 .. n_classes-1.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    if edges is None:
        edges = np.linspace(temperatures.min(), temperatures.max(), n_classes + 1)
    labels = np.digitize(temperatures, edges[1:-1])
    return np.clip(labels, 0, n_classes - 1).astype(int), np.asarray(edges)


def class_names(edges, short=False):
    fmt = "%.0f-%.0f" if short else "C%d: %.0f-%.0f °C"
    names = []
    for i in range(len(edges) - 1):
        if short:
            names.append(fmt % (edges[i], edges[i + 1]))
        else:
            names.append(fmt % (i, edges[i], edges[i + 1]))
    return names


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _block_split(indices, fractions, block_size, rng):
    """
    Deal contiguous blocks of `indices` into groups with the given fractions.

    Keeping whole blocks together stops neighbouring, nearly identical samples
    from landing on both sides of the split.
    """
    blocks = [indices[i:i + block_size] for i in range(0, len(indices), block_size)]
    order = rng.permutation(len(blocks))

    groups = [[] for _ in fractions]
    edges = np.cumsum(fractions) * len(blocks)
    for rank, block_id in enumerate(order):
        group = int(np.searchsorted(edges, rank, side="right"))
        group = min(group, len(groups) - 1)
        groups[group].append(blocks[block_id])

    out = []
    for group in groups:
        out.append(np.sort(np.concatenate(group)) if group
                   else np.array([], dtype=int))
    return out


def make_split(data, mode="ramp", block_size=25, val_fraction=0.2, seed=0):
    """Return (idx_train, idx_val, idx_test) as integer index arrays."""
    rng = np.random.default_rng(seed)
    n = len(data)
    all_idx = np.arange(n)

    if mode == "ramp":
        ramp_end = find_ramp_end(data)
        pool = all_idx[:ramp_end]
        idx_test = all_idx[ramp_end:]
        idx_train, idx_val = _block_split(
            pool, [1.0 - val_fraction, val_fraction], block_size, rng)

    elif mode == "blocked":
        idx_train, idx_val, idx_test = _block_split(
            all_idx, [0.64, 0.16, 0.20], block_size, rng)

    elif mode == "random":
        shuffled = rng.permutation(all_idx)
        n_train = int(0.64 * n)
        n_val = int(0.16 * n)
        idx_train = np.sort(shuffled[:n_train])
        idx_val = np.sort(shuffled[n_train:n_train + n_val])
        idx_test = np.sort(shuffled[n_train + n_val:])

    else:
        raise ValueError("unknown split mode %r" % mode)

    return idx_train, idx_val, idx_test


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

class MinMaxToUnit(object):
    """
    Scale each feature to [-1, 1] - the natural domain of the Chebyshev and
    Legendre synapse polynomials.

    Fitted on training data only.  Values outside the training range are
    clipped, which also acts as a physical saturation limit and stops the
    polynomials from being evaluated far outside the region where they were
    fitted.
    """

    def __init__(self, clip=1.25):
        self.clip = clip
        self.lo = None
        self.hi = None

    def fit(self, X):
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)
        span = self.hi - self.lo
        span[span == 0] = 1.0
        self._span = span
        return self

    def transform(self, X):
        scaled = 2.0 * (X - self.lo) / self._span - 1.0
        return np.clip(scaled, -self.clip, self.clip)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def out_of_range_fraction(self, X):
        scaled = 2.0 * (X - self.lo) / self._span - 1.0
        return float(np.mean(np.abs(scaled) > self.clip))


class StandardiseTarget(object):
    """Zero-mean / unit-variance scaling for the regression target."""

    def fit(self, y):
        self.mean = float(np.mean(y))
        self.std = float(np.std(y)) or 1.0
        return self

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse_transform(self, z):
        return z * self.std + self.mean


# ---------------------------------------------------------------------------
# One-call preparation
# ---------------------------------------------------------------------------

def prepare(split=None, n_classes=None, seed=None, path=None,
            block_size=None, val_fraction=None):
    """
    Load, label, split and scale in one go.

    Returns a Dataset with, for each of train/val/test:
        X_*            scaled currents, shape (n, 4), values in [-1, 1]
        T_*            temperature in °C
        y_reg_*        standardised temperature, shape (n, 1)
        y_class_*      integer class labels
        idx_*          row indices into the raw DataFrame
    plus the raw frame, the scalers, the class edges and their names.
    """
    split = config.SPLIT if split is None else split
    n_classes = config.N_CLASSES if n_classes is None else n_classes
    seed = config.SEED if seed is None else seed
    block_size = config.BLOCK_SIZE if block_size is None else block_size
    val_fraction = config.VAL_FRACTION if val_fraction is None else val_fraction

    data, load_info = load_raw(path, return_info=True)
    X_raw = data[config.CURRENT_COLUMNS].values.astype(float)
    T = data[config.TEMP_COLUMN].values.astype(float)

    # Class edges come from the full measured range so that the class
    # definition does not depend on the split.
    y_class_all, edges = make_temperature_classes(T, n_classes)

    idx_train, idx_val, idx_test = make_split(
        data, split, block_size, val_fraction, seed)

    scaler = MinMaxToUnit().fit(X_raw[idx_train])
    target_scaler = StandardiseTarget().fit(T[idx_train])

    clipped = np.mean([
        scaler.out_of_range_fraction(X_raw[idx_val]),
        scaler.out_of_range_fraction(X_raw[idx_test]),
    ])

    fields = dict(
        raw=data,
        X_raw=X_raw,
        T_all=T,
        split=split,
        n_classes=n_classes,
        class_edges=edges,
        class_names=class_names(edges),
        class_names_short=class_names(edges, short=True),
        scaler=scaler,
        target_scaler=target_scaler,
        heating=phase_mask(data),
        ramp_end=find_ramp_end(data),
        clipped_fraction=clipped,
        n_dropouts=load_info["n_dropouts"],
        dropout_temperatures=load_info["dropout_temperatures"],
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )

    for name, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        fields["X_" + name] = scaler.transform(X_raw[idx])
        fields["T_" + name] = T[idx]
        fields["y_reg_" + name] = target_scaler.transform(T[idx]).reshape(-1, 1)
        fields["y_class_" + name] = y_class_all[idx]

    return Dataset(**fields)


if __name__ == "__main__":
    for mode in ("ramp", "blocked", "random"):
        ds = prepare(split=mode)
        print(ds.describe())
        print()
    print("class definition (%d classes):" % ds.n_classes)
    for name in ds.class_names:
        print("   ", name)
