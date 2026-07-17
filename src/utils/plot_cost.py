"""Cost-to-capability across the pipeline — the project's headline thesis.

Per-stage compute cost, from real wall-clock:
  pretrain : 15.37 h on A100-40GB  -> $32.28  (run_card.json, logged cost)
  SFT      : 2562 s on L4          (wandb run seer-124m-sft-alpaca)
  DPO      : 6605 s on L4          (wandb run seer-dpo / dpo_log train_runtime)

L4 billed at Modal's list rate; the A100 pretrain cost is taken directly from
the run card. The point of the figure: pretraining is ~94% of the total spend —
post-training alignment is nearly free by comparison.
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
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)

L4_RATE = 0.80  # USD/hr, Modal L4 list price
STAGE_C = {"Pretrain": "#1f4e79", "SFT": "#3d6fa5", "DPO": "#c1432e"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results/figures/cost_per_stage"))
    args = p.parse_args()

    card = json.load(open(ROOT / "results/runs/seer_124m/run_card.json"))
    pre_cost = card["cost_usd"]
    pre_hr = card["duration_hr"]

    stages = [
        {"name": "Pretrain", "hr": pre_hr, "gpu": "A100-40GB", "cost": pre_cost},
        {"name": "SFT", "hr": 2562 / 3600, "gpu": "L4", "cost": 2562 / 3600 * L4_RATE},
        {"name": "DPO", "hr": 6605 / 3600, "gpu": "L4", "cost": 6605 / 3600 * L4_RATE},
    ]
    total = sum(s["cost"] for s in stages)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- panel A: absolute cost per stage (log-y; spend spans 2 orders) ---
    names = [s["name"] for s in stages]
    costs = [s["cost"] for s in stages]
    bars = ax1.bar(names, costs, color=[STAGE_C[n] for n in names], width=0.6)
    ax1.set_yscale("log")
    ax1.set_ylim(0.3, 60)
    ax1.set_ylabel("compute cost (USD, log)")
    ax1.set_title("Cost per stage", fontsize=11, fontweight="bold", loc="left")
    for bar, s in zip(bars, stages):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            s["cost"] * 1.12,
            f"${s['cost']:.2f}\n{s['hr']:.1f} h · {s['gpu']}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    # --- panel B: share of total spend (stacked bar) ---------------------
    left = 0
    for s in stages:
        frac = s["cost"] / total
        ax2.barh(0, frac, left=left, color=STAGE_C[s["name"]], height=0.5)
        if frac > 0.03:
            ax2.text(
                left + frac / 2,
                0,
                f"{s['name']}\n{frac * 100:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        left += frac
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("share of total pipeline cost")
    ax2.set_title(
        f"Where the ${total:.0f} went", fontsize=11, fontweight="bold", loc="left"
    )
    ax2.grid(False)

    fig.suptitle(
        f"Seer 124M — full-pipeline cost  (${total:.2f} total, "
        "pretraining dominates)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(
        f"wrote {out}.png and .pdf  "
        f"(pretrain ${pre_cost:.2f}, SFT ${stages[1]['cost']:.2f}, "
        f"DPO ${stages[2]['cost']:.2f}, total ${total:.2f})"
    )


if __name__ == "__main__":
    main()
