"""Plot hardware utilization during pretraining from train_log.jsonl.

The infra story behind the loss curve: how well the A100 was actually used.
  A: MFU (model FLOPs utilization) vs tokens
  B: throughput (tokens/sec) vs tokens
  C: GPU power draw vs tokens (with TDP reference)
  D: GPU temperature vs tokens
"""

import argparse
import json
import statistics
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
        "lines.linewidth": 1.6,
    }
)
LINE_C = "#1f4e79"
ACCENT_C = "#c1432e"
REF_C = "#888888"
A100_TDP_W = 400  # A100-SXM4-40GB thermal design power


def load_rows(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if r.get("step", 0) > 5]  # drop startup outliers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=str(ROOT / "results/runs/seer_124m/train_log.jsonl"))
    p.add_argument("--out", default=str(ROOT / "results/figures/hardware_util"))
    args = p.parse_args()

    rows = load_rows(args.log)
    # logged 'tokens' is 10x low (incremented per log, not per step); reconstruct
    # the honest count from step * total_batch_size, matching plot_training.py.
    card = json.load(open(Path(args.log).parent / "run_card.json"))
    tbs = card["cfg"]["train"]["total_batch_size"]
    toks = [r["step"] * tbs / 1e9 for r in rows]  # billions

    def col(k):
        return [r[k] for r in rows]

    mfu = [m * 100 for m in col("mfu")]
    tps = [t / 1e3 for t in col("tok_per_sec")]  # k tok/s
    power = col("gpu_power_w")
    temp = col("gpu_temp_c")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    med_mfu = statistics.median(mfu)
    ax1.plot(toks, mfu, color=LINE_C)
    ax1.axhline(med_mfu, color=ACCENT_C, ls="--", lw=1)
    ax1.annotate(
        f"median {med_mfu:.0f}%",
        xy=(toks[len(toks) // 2], med_mfu),
        xytext=(toks[-1] * 0.55, med_mfu + 4),
        color=ACCENT_C,
        fontsize=9,
    )
    ax1.set_ylim(0, 60)
    ax1.set_xlabel("training tokens (billions)")
    ax1.set_ylabel("MFU (%)")
    ax1.set_title(
        "Model FLOPs utilization", fontsize=11, fontweight="bold", loc="left"
    )

    ax2.plot(toks, tps, color=LINE_C)
    ax2.set_xlabel("training tokens (billions)")
    ax2.set_ylabel("throughput (k tokens/sec)")
    ax2.set_title("Throughput", fontsize=11, fontweight="bold", loc="left")

    ax3.plot(toks, power, color=LINE_C, label="draw")
    ax3.axhline(A100_TDP_W, color=REF_C, ls="--", lw=1, label=f"TDP ({A100_TDP_W} W)")
    ax3.set_ylim(300, A100_TDP_W + 30)
    ax3.set_xlabel("training tokens (billions)")
    ax3.set_ylabel("GPU power (W)")
    ax3.set_title("Power draw", fontsize=11, fontweight="bold", loc="left")
    ax3.legend(fontsize=8, loc="lower right")

    ax4.plot(toks, temp, color=LINE_C)
    ax4.set_xlabel("training tokens (billions)")
    ax4.set_ylabel("GPU temperature (°C)")
    ax4.set_title("Temperature", fontsize=11, fontweight="bold", loc="left")

    fig.suptitle(
        "Seer pretraining — A100-40GB utilization  "
        "(GPU pinned at 100%, ~185k tok/s, 51% MFU)",
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
