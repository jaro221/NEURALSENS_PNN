# -*- coding: utf-8 -*-
"""
An overview image of the measurement itself, before any modelling.

Three panels, sharing one story:

  top     the temperature programme - 30 to 150 °C in 10 °C steps, then a
          free cooling curve back to ~31 °C
  middle  the four device currents through the same run, with the read-out
          dropouts marked
  bottom  current against temperature, one panel per device, heating separated
          from cooling so the thermal hysteresis is visible

Design notes
------------
The four devices are an *ordered* quantity (rising bias voltage), so they get a
single-hue ordinal ramp, light to dark, rather than four unrelated colours.
Heating and cooling are two identities, so they get two categorical hues.
No panel uses two y-scales: the temperature and the currents are stacked and
share the time axis instead of being folded onto one plot with twin axes.

Currents are drawn on a symmetric-log axis.  The readings span two decades and
27 samples go non-positive on at least one channel, so a plain log axis would
silently drop them - and those samples are the clearest evidence of the
read-out faults, which is exactly what this figure is meant to show.

Run directly (F5 in Spyder).  Writes results/00_data_overview.png.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import config
import data_loader

# --- palette ---------------------------------------------------------------
# Ordinal ramp for the four bias voltages (validated: monotone lightness,
# single hue, light end clears the surface).
DEVICE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
HEATING = "#eb6834"
COOLING = "#1baf7a"
DROPOUT = "#d03b3b"          # status: bad data, always paired with a label

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

OUTPUT = "00_data_overview.png"


def style_axes(ax):
    """Recessive chrome: hairline solid grid behind the data, no box."""
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=4, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)


def contiguous_runs(indices, max_gap=3):
    """Group sorted indices into runs, so dropout bursts can be shaded."""
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > max_gap)[0] + 1
    return [run for run in np.split(indices, breaks) if len(run)]


def main():
    config.ensure_results_dir()

    # Load without cleaning, so the dropouts can be shown rather than hidden.
    data = data_loader.load_raw(remove_dropouts=False)
    dropout_mask = data_loader.find_dropouts(data)
    heating = data_loader.phase_mask(data)

    time_min = data["time"].values / 60.0
    set_temp = data["set_temp"].values
    act_temp = data[config.TEMP_COLUMN].values
    currents = data[config.CURRENT_COLUMNS].values * 1e9      # nA
    switch = time_min[data_loader.find_ramp_end(data)]

    fig = plt.figure(figsize=(13.5, 13.0), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.35, 1.25],
                            hspace=0.42, wspace=0.30,
                            left=0.07, right=0.975, top=0.915, bottom=0.055)

    # ---------------------------------------------------------------
    # top - the temperature programme
    # ---------------------------------------------------------------
    ax_t = fig.add_subplot(grid[0, :])
    style_axes(ax_t)
    ax_t.step(time_min, set_temp, where="post", color=MUTED, linewidth=1.6,
              label="set point")
    ax_t.plot(time_min, act_temp, color=INK, linewidth=2.0,
              label="measured plate temperature")
    ax_t.axvline(switch, color=AXIS, linewidth=1.2)
    ax_t.set_ylabel("temperature [°C]", color=INK_2, fontsize=10)
    ax_t.set_xlim(0, time_min.max())
    ax_t.set_ylim(20, 165)
    ax_t.legend(loc="upper left", frameon=False, fontsize=9,
                labelcolor=INK_2, ncol=2, columnspacing=1.4)
    ax_t.set_title("Temperature programme", loc="left", fontsize=12,
                   fontweight="bold", color=INK, pad=8)

    for x, text in ((switch / 2, "heating ramp"),
                    ((switch + time_min.max()) / 2, "free cooling")):
        ax_t.text(x, 30, text, ha="center", va="bottom", fontsize=9.5,
                  color=INK_2, fontweight="bold")

    # ---------------------------------------------------------------
    # middle - the four currents through the run
    # ---------------------------------------------------------------
    ax_i = fig.add_subplot(grid[1, :], sharex=ax_t)
    style_axes(ax_i)

    runs = contiguous_runs(np.where(dropout_mask)[0])
    for k, run in enumerate(runs):
        lo = time_min[run[0]] - 0.3
        hi = time_min[run[-1]] + 0.3
        ax_i.axvspan(lo, hi, color=DROPOUT, alpha=0.16, linewidth=0,
                     label="read-out dropout (removed)" if k == 0 else None)

    for i, (column, colour) in enumerate(zip(config.CURRENT_COLUMNS, DEVICE_RAMP)):
        ax_i.plot(time_min, currents[:, i], color=colour, linewidth=1.5,
                  label="%.1f V" % config.BIAS_VOLTAGES[i])
        # selective direct label at the right edge, not a value on every point
        ax_i.annotate(" D%d @ %.1f V" % (i + 1, config.BIAS_VOLTAGES[i]),
                      xy=(time_min[-1], currents[-1, i]),
                      xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=8.5, color=colour,
                      fontweight="bold")

    ax_i.axvline(switch, color=AXIS, linewidth=1.2)
    ax_i.set_yscale("symlog", linthresh=100)
    ax_i.set_yticks([-100, 0, 100, 1000, 10000])
    ax_i.set_yticklabels(["-100", "0", "100", "1k", "10k"])
    ax_i.axhspan(-100, 100, color=GRID, alpha=0.45, linewidth=0, zorder=0)
    ax_i.text(time_min.max() * 0.012, 0, "linear below 100 nA",
              fontsize=8, color=MUTED, va="center")
    ax_i.set_ylabel("current [nA]   (symmetric log)", color=INK_2, fontsize=10)
    ax_i.set_xlabel("time [min]", color=INK_2, fontsize=10)
    ax_i.set_xlim(0, time_min.max() * 1.075)
    ax_i.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005),
                frameon=False, fontsize=9, ncol=5, labelcolor=INK_2,
                columnspacing=1.6, borderaxespad=0.0)
    ax_i.set_title("Device currents", loc="left", fontsize=12,
                   fontweight="bold", color=INK, pad=26)

    # ---------------------------------------------------------------
    # bottom - current against temperature, one device per panel
    # ---------------------------------------------------------------
    keep = ~dropout_mask
    for i, column in enumerate(config.CURRENT_COLUMNS):
        ax = fig.add_subplot(grid[2, i])
        style_axes(ax)
        for mask, colour, name in ((heating & keep, HEATING, "heating"),
                                   (~heating & keep, COOLING, "cooling")):
            ax.scatter(act_temp[mask], currents[mask, i], s=9, alpha=0.55,
                       color=colour, linewidths=0, label=name)
        ax.set_title("D%d @ %.1f V" % (i + 1, config.BIAS_VOLTAGES[i]),
                     fontsize=10.5, fontweight="bold", color=INK, pad=6)
        ax.set_xlabel("temperature [°C]", color=INK_2, fontsize=9.5)
        if i == 0:
            ax.set_ylabel("current [nA]", color=INK_2, fontsize=10)
            ax.legend(loc="upper left", frameon=False, fontsize=9,
                      labelcolor=INK_2, handletextpad=0.4)
        ax.set_xlim(25, 155)

    # ---------------------------------------------------------------
    fig.suptitle("Temperature sweep of four diodes at four bias points",
                 x=0.07, y=0.975, ha="left", fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.07, 0.945,
             "%d samples, one every 2 s   |   30-150 °C in 10 °C steps, "
             "then free cooling   |   %d read-out dropouts marked and excluded "
             "from the lower panels"
             % (len(data), int(dropout_mask.sum())),
             ha="left", fontsize=10, color=INK_2)

    path = config.result_path(OUTPUT)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("written: %s" % path)
    plt.show()
    return fig


if __name__ == "__main__":
    figure = main()
