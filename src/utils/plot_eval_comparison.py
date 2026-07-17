"""Three-way eval comparison: base -> SFT -> DPO, as a dumbbell per task.

Walks every lm-eval results_*.json under results/lm_eval, keeps the full-set
raw run for each (model, task) (largest n-samples), and renders two panels:
  left  — capability (expected to hold steady; a small drop = alignment tax)
  right — alignment & instruction (the metrics post-training should move)
Chance baselines are drawn per task so near-random scores are visible as such.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

mpl.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)

# task -> (metric prefix, label, bucket, chance)
TASKS = {
    "hellaswag": ("acc_norm", "HellaSwag", "capability", 0.25),
    "arc_easy": ("acc_norm", "ARC-easy", "capability", 0.25),
    "arc_challenge": ("acc_norm", "ARC-challenge", "capability", 0.25),
    "piqa": ("acc_norm", "PIQA", "capability", 0.50),
    "winogrande": ("acc", "WinoGrande", "capability", 0.50),
    "lambada_openai": ("acc", "LAMBADA", "capability", 0.0),
    "truthfulqa_mc2": ("acc", "TruthfulQA", "alignment", None),
    "ifeval": ("prompt_level_strict_acc", "IFEval", "alignment", 0.0),
}
MODELS = ["base", "sft", "dpo"]
STAGE_C = {"base": "#9bb8d3", "sft": "#3d6fa5", "dpo": "#c1432e"}
STAGE_LABEL = {"base": "Base", "sft": "SFT", "dpo": "DPO"}


def get_metric(res, pfx):
    val = err = None
    for k, v in res.items():
        if not isinstance(v, (int, float)):
            continue
        if k.startswith(pfx + "_stderr,"):
            err = v
        elif k.startswith(pfx + ","):
            val = v
    return val, err


def model_of(path):
    """Infer model name from a path like .../evals_full/sft/... ."""
    parts = Path(path).parts
    for m in MODELS:
        if m in parts:
            return m
    return None


def load(root):
    """(model, task) -> (value, stderr, n). Keeps the largest-n raw run."""
    best = {}
    for fp in glob.glob(os.path.join(root, "**", "results_*.json"), recursive=True):
        if "evals_chat" in fp:  # templated runs are a separate experiment
            continue
        model = model_of(fp)
        if model is None:
            continue
        try:
            blob = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        results = blob.get("results", {})
        nsamp = blob.get("n-samples", {})
        for task, (pfx, *_rest) in TASKS.items():
            if task not in results:
                continue
            val, err = get_metric(results[task], pfx)
            if val is None:
                continue
            n = nsamp.get(task, {}).get("effective", 0) or 0
            key = (model, task)
            if key not in best or n > best[key][2]:
                best[key] = (val, err, n)
    return best


def panel(ax, data, tasks, title):
    labels = [TASKS[t][1] for t in tasks]
    y = list(range(len(tasks)))
    for yi, task in zip(y, tasks):
        chance = TASKS[task][3]
        if chance is not None:
            ax.plot(
                [chance * 100, chance * 100],
                [yi - 0.32, yi + 0.32],
                color="#999",
                lw=1.2,
                ls=(0, (2, 2)),
                zorder=1,
            )
        pts = []
        for m in MODELS:
            if (m, task) in data:
                v, e, _n = data[(m, task)]
                pts.append((m, v * 100, (e or 0) * 100))
        if len(pts) >= 2:  # connecting line base->dpo
            xs = [p[1] for p in pts]
            ax.plot([min(xs), max(xs)], [yi, yi], color="#c9c7bf", lw=2.5, zorder=2)
        for m, v, e in pts:
            ax.errorbar(
                v, yi, xerr=e, fmt="o", ms=8, color=STAGE_C[m],
                ecolor=STAGE_C[m], elinewidth=1, capsize=2, zorder=3,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 70)
    ax.set_xlabel("score (%)")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT / "results/lm_eval"))
    p.add_argument("--out", default=str(ROOT / "results/figures/eval_comparison"))
    args = p.parse_args()

    data = load(args.root)
    if not data:
        raise SystemExit(f"no results under {args.root}")

    cap = [t for t in TASKS if TASKS[t][2] == "capability"]
    align = [t for t in TASKS if TASKS[t][2] == "alignment"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 1.2]}
    )
    panel(ax1, data, cap, "Capability  (expected: hold steady)")
    panel(ax2, data, align, "Alignment & instruction")

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8, color=STAGE_C[m],
                   label=STAGE_LABEL[m])
        for m in MODELS
    ]
    handles.append(
        plt.Line2D([], [], color="#999", lw=1.2, ls=(0, (2, 2)), label="chance")
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Seer 124M — benchmarks across post-training  "
        "(full sets, 0-shot, ± stderr)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.png and .pdf")
    for m in MODELS:
        row = {TASKS[t][1]: round(data[(m, t)][0] * 100, 1)
               for t in TASKS if (m, t) in data}
        print(f"  {m}: {row}")


if __name__ == "__main__":
    main()
