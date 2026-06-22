"""Render benchmark comparison table from eval results.



Run: uv run python src/utils/plot_evals.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "lm_eval" / "summary.json"
FIG_DIR = ROOT / "results" / "figures"

HERO_FILL = "#FAECE7"  # coral 50
HERO_TEXT = "#4A1B0C"  # coral 900
SHOW_RANDOM = False

mpl.rcParams.update({"savefig.dpi": 300, "font.family": "DejaVu Sans"})

PRETTY = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-easy",
    "arc_challenge": "ARC-challenge",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "lambada_openai": "LAMBADA",
    "ifeval": "IFEval",
}


def main():
    spec = json.loads(SUMMARY.read_text())
    tasks = spec["tasks"]
    models = spec["models"]
    names = list(models)
    cols = names + (["Random"] if SHOW_RANDOM else [])

    # flat render list with section markers
    rows = []
    last_section = None
    for task, meta in tasks.items():
        section = meta.get("section")
        if section != last_section:
            rows.append(("section", section))
            last_section = section
        rows.append(("task", task, meta))

    n_rows = len(rows) + 1  # + header
    n_val = len(cols)

    label_w = 0.30
    val_w = (1 - label_w) / n_val
    row_h = 1.0 / n_rows

    fig_w = 3.2 + 1.7 * n_val
    fig_h = 0.55 * n_rows + 0.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def col_x(j):
        return label_w + j * val_w + val_w / 2

    def row_y(i):  # i=0 is header
        return 1 - i * row_h - row_h / 2

    # hero column shading (behind everything)
    hx = label_w
    ax.add_patch(
        FancyBboxPatch(
            (hx + 0.004, 0.004),
            val_w - 0.008,
            1 - 0.008,
            boxstyle="round,pad=0,rounding_size=0.02",
            facecolor=HERO_FILL,
            edgecolor="none",
            zorder=0,
        )
    )

    # header
    for j, c in enumerate(cols):
        ax.text(
            col_x(j),
            row_y(0),
            c,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=HERO_TEXT if j == 0 else "#5F5E5A",
        )
    ax.plot([0, 1], [1 - row_h, 1 - row_h], color="#B4B2A9", lw=0.8, zorder=1)

    for r, item in enumerate(rows, start=1):
        y = row_y(r)
        if item[0] == "section":
            ax.plot(
                [0, 1],
                [1 - r * row_h, 1 - r * row_h],
                color="#B4B2A9",
                lw=0.8,
                zorder=1,
            )
            ax.text(
                0.01, y, item[1], ha="left", va="center", fontsize=9.5, color="#888780"
            )
            continue

        _, task, meta = item
        ax.text(
            0.01,
            y + row_h * 0.12,
            PRETTY.get(task, task),
            ha="left",
            va="center",
            fontsize=11,
            color="#2C2C2A",
        )
        ax.text(
            0.01,
            y - row_h * 0.24,
            meta["metric"],
            ha="left",
            va="center",
            fontsize=8.5,
            color="#888780",
        )

        scores = {n: models[n].get(task) for n in names}
        best = max((v for v in scores.values() if v is not None), default=None)

        for j, c in enumerate(cols):
            if c == "Random":
                v = meta.get("random")
                if v is None:
                    txt = "-"
                elif v == 0:
                    txt = "~0%"
                else:
                    txt = f"{v * 100:.1f}%"
                ax.text(
                    col_x(j),
                    y,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#B4B2A9",
                )
                continue
            v = scores[c]
            if v is None:
                ax.text(
                    col_x(j),
                    y,
                    "-",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#B4B2A9",
                )
                continue
            is_best = best is not None and abs(v - best) < 1e-9
            ax.text(
                col_x(j),
                y,
                f"{v * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold" if is_best else "normal",
                color=HERO_TEXT if j == 0 else "#2C2C2A",
            )

        ax.plot(
            [0, 1],
            [1 - (r + 1) * row_h, 1 - (r + 1) * row_h],
            color="#E6E4DC",
            lw=0.6,
            zorder=1,
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"eval_table.{ext}", bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'eval_table.png'} and .pdf")


if __name__ == "__main__":
    main()
