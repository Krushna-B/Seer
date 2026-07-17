"""Plot SFT training dynamics from the HF trainer_state.json (log_history).

Four panels, the SFT analog of the DPO curve:
  A: loss vs step            (train smoothed + eval points)
  B: next-token accuracy     (train + eval; the SFT "it's learning" line)
  C: LR schedule + grad norm (warmup->cosine on a twin axis with stability)
  D: entropy                 (train + eval; distribution sharpening as it fits)
"""

import argparse
import json
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
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    }
)
TRAIN_C = "#9bb8d3"  # muted blue (raw)
TRAIN_S = "#1f4e79"  # strong blue (smoothed)
VAL_C = "#c1432e"  # red (eval)
ACCENT_C = "#5a9367"  # green (grad norm)


def load_rows(path):
    """Accept either the HF trainer_state.json or a raw jsonl log."""
    text = Path(path).read_text()
    try:
        blob = json.loads(text)
        return blob["log_history"]
    except (json.JSONDecodeError, KeyError):
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def rolling_mean(xs, window):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def series(rows, key):
    """(steps, values) for rows that carry `key`."""
    pts = [(r["step"], r[key]) for r in rows if key in r and "step" in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log",
        default=str(ROOT / "artifacts/sft_out/checkpoint-1500/trainer_state.json"),
    )
    p.add_argument("--out", default=str(ROOT / "results/figures/sft_curves"))
    args = p.parse_args()

    rows = load_rows(args.log)

    l_step, loss = series(rows, "loss")
    evl_step, evl = series(rows, "eval_loss")
    a_step, acc = series(rows, "mean_token_accuracy")
    eva_step, eva = series(rows, "eval_mean_token_accuracy")
    lr_step, lr = series(rows, "learning_rate")
    g_step, gnorm = series(rows, "grad_norm")
    e_step, ent = series(rows, "entropy")
    eve_step, eve = series(rows, "eval_entropy")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    # --- panel A: loss ---------------------------------------------------
    ax1.plot(l_step, loss, color=TRAIN_C, lw=0.8, alpha=0.6, label="train (raw)")
    ax1.plot(l_step, rolling_mean(loss, 8), color=TRAIN_S, label="train (smoothed)")
    if evl_step:
        ax1.plot(evl_step, evl, color=VAL_C, marker="o", ms=3, lw=1.4, label="eval")
        ax1.annotate(
            f"{evl[-1]:.2f}",
            xy=(evl_step[-1], evl[-1]),
            xytext=(evl_step[-1] * 0.78, evl[-1] + 0.03),
            color=VAL_C,
            fontsize=9,
        )
    ax1.set_xlabel("training step")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("SFT loss", fontsize=11, fontweight="bold", loc="left")
    ax1.legend(fontsize=8)

    # --- panel B: next-token accuracy (the "it's learning" line) ---------
    ax2.plot(a_step, acc, color=TRAIN_C, lw=0.8, alpha=0.6, label="train (raw)")
    ax2.plot(a_step, rolling_mean(acc, 8), color=TRAIN_S, label="train (smoothed)")
    if eva_step:
        ax2.plot(eva_step, eva, color=VAL_C, marker="o", ms=3, lw=1.4, label="eval")
        ax2.annotate(
            f"{eva[-1]:.2f}",
            xy=(eva_step[-1], eva[-1]),
            xytext=(eva_step[-1] * 0.78, eva[-1] + 0.004),
            color=VAL_C,
            fontsize=9,
        )
    ax2.set_xlabel("training step")
    ax2.set_ylabel("next-token accuracy")
    ax2.set_title("Token accuracy", fontsize=11, fontweight="bold", loc="left")
    ax2.legend(fontsize=8)

    # --- panel C: LR schedule + grad norm on a twin axis -----------------
    ax3.plot(lr_step, lr, color=TRAIN_S, label="learning rate")
    ax3.set_xlabel("training step")
    ax3.set_ylabel("learning rate", color=TRAIN_S)
    ax3.tick_params(axis="y", labelcolor=TRAIN_S)
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax3.set_title(
        "LR schedule + grad norm", fontsize=11, fontweight="bold", loc="left"
    )

    ax3b = ax3.twinx()
    ax3b.spines["top"].set_visible(False)
    ax3b.grid(False)
    ax3b.plot(g_step, gnorm, color=ACCENT_C, lw=1.0, alpha=0.8, label="grad norm")
    ax3b.set_ylabel("grad norm", color=ACCENT_C)
    ax3b.tick_params(axis="y", labelcolor=ACCENT_C)

    lines = ax3.get_lines() + ax3b.get_lines()
    ax3.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")

    # --- panel D: entropy (distribution sharpening as it fits) -----------
    ax4.plot(e_step, ent, color=TRAIN_C, lw=0.8, alpha=0.6, label="train (raw)")
    ax4.plot(e_step, rolling_mean(ent, 8), color=TRAIN_S, label="train (smoothed)")
    if eve_step:
        ax4.plot(eve_step, eve, color=VAL_C, marker="o", ms=3, lw=1.4, label="eval")
    ax4.set_xlabel("training step")
    ax4.set_ylabel("predictive entropy (nats)")
    ax4.set_title("Entropy", fontsize=11, fontweight="bold", loc="left")
    ax4.legend(fontsize=8)

    fig.suptitle(
        "Seer SFT — instruction tuning on Alpaca (base checkpoint)",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.png and .pdf")


if __name__ == "__main__":
    main()
