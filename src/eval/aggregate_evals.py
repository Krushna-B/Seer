"""Aggregate lm-eval-harness JSON outputs into a tidy CSV + comparison figure.

Walks the eval output root (one subdir per checkpoint, e.g. base/ sft/ dpo/),
pulls the headline metric per task, writes eval_summary.csv, and renders a
grouped-bar chart comparing checkpoints across all metrics (with stderr bars).
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# task -> (lm-eval metric prefix, nice label, bucket).  acc_norm for the
# length-normalized multiple-choice tasks (the numbers published for GPT-2).
METRICS = {
    "hellaswag": ("acc_norm", "HellaSwag", "capability"),
    "arc_challenge": ("acc_norm", "ARC-c", "capability"),
    "arc_easy": ("acc_norm", "ARC-e", "capability"),
    "piqa": ("acc_norm", "PIQA", "capability"),
    "winogrande": ("acc", "WinoGrande", "capability"),
    "lambada_openai": ("acc", "LAMBADA", "capability"),
    "mmlu": ("acc", "MMLU", "capability"),
    "ifeval": ("prompt_level_strict_acc", "IFEval", "instruction"),
    "truthfulqa_mc2": ("acc", "TruthfulQA", "alignment"),
}

# preferred plotting order for checkpoints; unknown names appended after
CKPT_ORDER = ["base", "sft", "dpo"]
PALETTE = ["#9bb8d3", "#1f4e79", "#c1432e", "#5a9367", "#d4a017"]

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


def get_metric(task_res, prefix):
    """lm-eval keys look like 'acc_norm,none' / 'acc_norm_stderr,none'."""
    val = err = None
    for k, v in task_res.items():
        if not isinstance(v, (int, float)):
            continue
        if k.startswith(prefix + "_stderr,"):
            err = v
        elif k.startswith(prefix + ","):
            val = v
    return val, err


def load_all(root):
    """name -> {label: (value, stderr, bucket)} across all results*.json."""
    data = defaultdict(dict)
    for fp in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        try:
            blob = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        if "results" not in blob:
            continue
        name = os.path.relpath(fp, root).split(os.sep)[0]
        for task, res in blob["results"].items():
            if task not in METRICS:
                continue
            prefix, label, bucket = METRICS[task]
            val, err = get_metric(res, prefix)
            if val is not None:
                data[name][label] = (val, err, bucket)
    return data


def ordered_names(data):
    known = [n for n in CKPT_ORDER if n in data]
    return known + sorted(n for n in data if n not in CKPT_ORDER)


def ordered_labels(data):
    labels = []
    for _task, (_p, label, _b) in METRICS.items():
        if label not in labels and any(label in data[n] for n in data):
            labels.append(label)
    return labels


def write_csv(data, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "metric", "bucket", "value", "stderr"])
        for name in ordered_names(data):
            for label, (val, err, bucket) in data[name].items():
                w.writerow([name, label, bucket, f"{val:.4f}", f"{err:.4f}" if err else ""])
    print(f"wrote {out_path}")


def plot(data, out_path):
    names = ordered_names(data)
    labels = ordered_labels(data)
    x = np.arange(len(labels))
    w = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(labels)), 4.6))
    for i, name in enumerate(names):
        vals = [data[name].get(lbl, (0, 0, ""))[0] for lbl in labels]
        errs = [data[name].get(lbl, (0, 0, ""))[1] or 0 for lbl in labels]
        ax.bar(
            x - 0.4 + w / 2 + i * w,
            vals,
            w,
            yerr=errs,
            capsize=2,
            label=name,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Seer eval: base → SFT → DPO", fontweight="bold", loc="left"
    )
    ax.legend(ncol=len(names), loc="upper right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight")
    print(f"wrote {out_path}.png and .pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="eval output root")
    p.add_argument("--out", required=True, help="dir for csv + figure")
    args = p.parse_args()

    data = load_all(args.root)
    if not data:
        raise SystemExit(f"no parseable results under {args.root}")
    print("found checkpoints:", ", ".join(ordered_names(data)))

    os.makedirs(args.out, exist_ok=True)
    write_csv(data, os.path.join(args.out, "eval_summary.csv"))
    plot(data, os.path.join(args.out, "eval_comparison"))


if __name__ == "__main__":
    main()
