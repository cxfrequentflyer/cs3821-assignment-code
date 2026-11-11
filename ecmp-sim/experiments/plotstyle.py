"""Shared figure style for the experiments (the same palette as exp01).

Categorical hues go in this fixed order and follow the entity, never its rank.
The palette passes the colorblind-separation checks; three hues are under 3:1
contrast on the surface, so every figure also direct-labels its lines and
every script prints the numbers as a table.
"""
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def line(ax, x, y, i, label, dashed=False, marker=True):
    """One series: 2px line, >= 8px markers with a surface ring, direct label at the end."""
    ax.plot(x, y, color=SERIES[i], lw=2, ls=(0, (4, 3)) if dashed else "-",
            marker="o" if marker else None, ms=7, mec=SURFACE, mew=2, label=label)
    if label:
        ax.annotate(label, (list(x)[-1], list(y)[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
