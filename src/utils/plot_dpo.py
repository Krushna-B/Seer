"""Plot DPO training dynamics from dpo_log.jsonl (TRL trainer.state.log_history).

Two panels, the DPO analog of the pretraining loss curve:
  A: reward accuracy vs step  (train smoothed + eval points, chance line at 0.5)
  B: chosen vs rejected rewards diverging, with the margin shaded between them
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
TRAIN_S = "#1f4e79"  # strong blue (smoothed / chosen)
VAL_C = "#c1432e"  # red (eval / rejected)
MARGIN_C = "#5a9367"  # green fill


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


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
    p.add_argument("--log", default=str(ROOT / "results/runs/seer-dpo/dpo_log.jsonl"))
    p.add_argument("--out", default=str(ROOT / "results/figures/dpo_curves"))
    args = p.parse_args()

    rows = load_rows(args.log)

    tr_step, tr_acc = series(rows, "rewards/accuracies")
    ev_step, ev_acc = series(rows, "eval_rewards/accuracies")
    c_step, chosen = series(rows, "rewards/chosen")
    _, rejected = series(rows, "rewards/rejected")
    m_step, margin = series(rows, "rewards/margins")
    evm_step, evm = series(rows, "eval_rewards/margins")
    l_step, loss = series(rows, "loss")
    evl_step, evl = series(rows, "eval_loss")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    # --- panel A: reward accuracy ----------------------------------------
    ax1.axhline(0.5, color="#888", ls="--", lw=1, label="chance (0.5)")
    ax1.plot(tr_step, tr_acc, color=TRAIN_C, lw=0.8, alpha=0.6, label="train (raw)")
    ax1.plot(
        tr_step, rolling_mean(tr_acc, 20), color=TRAIN_S, label="train (smoothed)"
    )
    if ev_step:
        ax1.plot(ev_step, ev_acc, color=VAL_C, marker="o", ms=3, lw=1.4, label="eval")
        ax1.annotate(
            f"{ev_acc[-1]:.2f}",
            xy=(ev_step[-1], ev_acc[-1]),
            xytext=(ev_step[-1] * 0.8, ev_acc[-1] + 0.03),
            color=VAL_C,
            fontsize=9,
        )
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("reward accuracy")
    ax1.set_title("Preference accuracy", fontsize=11, fontweight="bold", loc="left")
    ax1.legend(fontsize=8)

    # --- panel B: reward margin (the cleanest "it worked" line) ----------
    ax2.axhline(0, color="#888", ls="--", lw=0.8)
    ax2.plot(m_step, margin, color=TRAIN_C, lw=0.8, alpha=0.6, label="train (raw)")
    ax2.plot(m_step, rolling_mean(margin, 20), color=TRAIN_S, label="train (smoothed)")
    if evm_step:
        ax2.plot(evm_step, evm, color=VAL_C, marker="o", ms=3, lw=1.4, label="eval")
        ax2.annotate(
            f"{evm[-1]:.2f}",
            xy=(evm_step[-1], evm[-1]),
            xytext=(evm_step[-1] * 0.8, evm[-1] + 0.02),
            color=VAL_C,
            fontsize=9,
        )
    ax2.set_xlabel("training step")
    ax2.set_ylabel("reward margin (chosen − rejected)")
    ax2.set_title("Preference margin", fontsize=11, fontweight="bold", loc="left")
    ax2.legend(fontsize=8)

    # --- panel C: chosen vs rejected reward, margin shaded --------------
    ax3.plot(c_step, chosen, color=TRAIN_S, label="chosen")
    ax3.plot(c_step, rejected, color=VAL_C, label="rejected")
    ax3.fill_between(
        c_step, rejected, chosen, color=MARGIN_C, alpha=0.25, label="margin"
    )
    ax3.axhline(0, color="#888", ls="--", lw=0.8)
    ax3.set_xlabel("training step")
    ax3.set_ylabel("implicit reward")
    ax3.set_title("Reward separation", fontsize=11, fontweight="bold", loc="left")
    ax3.legend(fontsize=8)

    # --- panel D: loss (weak signal, shown honestly) -------------------
    ax4.plot(l_step, loss, color=TRAIN_C, lw=0.8, alpha=0.6, label="train (raw)")
    ax4.plot(l_step, rolling_mean(loss, 20), color=TRAIN_S, label="train (smoothed)")
    if evl_step:
        ax4.plot(evl_step, evl, color=VAL_C, marker="o", ms=3, lw=1.4, label="eval")
    ax4.axhline(0.693, color="#888", ls="--", lw=0.8, label="init (ln 2)")
    ax4.set_xlabel("training step")
    ax4.set_ylabel("DPO loss")
    ax4.set_title("Loss", fontsize=11, fontweight="bold", loc="left")
    ax4.legend(fontsize=8)

    fig.suptitle(
        "Seer DPO — preference alignment on the SFT checkpoint",
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
