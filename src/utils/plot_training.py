import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

RUN = "seer_124m"

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "results" / "runs" / RUN
FIG_DIR = ROOT / "results" / "figures"


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
TRAIN_C = "#9bb8d3"  # muted blue (raw, noisy)
TRAIN_S = "#1f4e79"  # strong blue (smoothed)
VAL_C = "#c1432e"  # red


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


def main():
    rows = load_rows(RUN_DIR / "train_log.jsonl")
    card = json.load(open(RUN_DIR / "run_card.json"))
    tbs = card["cfg"]["train"]["total_batch_size"]
    n_params = card["n_params"]

    # honest token axis (logged 'tokens' is 10x low)
    tokens_b = [r["step"] * tbs / 1e9 for r in rows]
    train_loss = [r["loss"] for r in rows]
    train_smooth = rolling_mean(train_loss, window=20)

    val = [
        (r["step"] * tbs / 1e9, r["cost_usd"], r["val_loss"])
        for r in rows
        if "val_loss" in r
    ]
    v_tok = [v[0] for v in val]
    v_cost = [v[1] for v in val]
    v_loss = [v[2] for v in val]
    best_val = min(v_loss)
    total_tokens_b = tokens_b[-1]
    total_cost = rows[-1]["cost_usd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- panel A: loss vs tokens (log-y shows the full 11 -> 3 descent) ----
    ax1.plot(
        tokens_b, train_loss, color=TRAIN_C, lw=0.8, alpha=0.6, label="train (raw)"
    )
    ax1.plot(tokens_b, train_smooth, color=TRAIN_S, label="train (smoothed)")
    ax1.plot(v_tok, v_loss, color=VAL_C, marker="o", ms=3, lw=1.4, label="validation")
    ax1.set_yscale("log")
    ax1.set_ylim(2.8, 12)
    ax1.set_yticks([3, 4, 5, 7, 10])
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.minorticks_off()
    ax1.set_xlabel("training tokens (billions)")
    ax1.set_ylabel("cross-entropy loss (log)")
    ax1.set_title("Pretraining loss", fontsize=11, fontweight="bold", loc="left")

    # zoom inset: the convergence region (final validation loss)
    axins = ax1.inset_axes([0.46, 0.42, 0.50, 0.46])
    zoom = [(t, l) for t, l in zip(v_tok, v_loss) if t >= 2.0]
    axins.plot(
        [z[0] for z in zoom],
        [z[1] for z in zoom],
        color=VAL_C,
        marker="o",
        ms=3,
        lw=1.4,
    )
    axins.annotate(
        f"{best_val:.2f}",
        xy=(total_tokens_b, best_val),
        xytext=(total_tokens_b * 0.62, best_val - 0.07),
        color=VAL_C,
        fontsize=8,
    )
    axins.set_ylim(2.95, 3.55)
    axins.set_xlim(2.0, total_tokens_b * 1.02)
    axins.grid(alpha=0.25)
    axins.tick_params(labelsize=7)
    axins.set_title("convergence", fontsize=8, color="#444")

    # --- panel B: validation loss vs cost ($) -----------------------------
    ax2.plot(v_cost, v_loss, color=VAL_C, marker="o", ms=3, lw=1.4)
    ax2.annotate(
        f"${total_cost:.0f}  ->  val {best_val:.2f}",
        xy=(total_cost, best_val),
        xytext=(total_cost * 0.28, best_val + 1.4),
        color=VAL_C,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=VAL_C, lw=1),
    )
    ax2.set_yscale("log")
    ax2.set_ylim(2.8, 12)
    ax2.set_yticks([3, 4, 5, 7, 10])
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.minorticks_off()
    ax2.set_xlabel("cumulative compute cost (USD)")
    ax2.set_ylabel("validation loss (log)")
    ax2.set_title("Cost to capability", fontsize=11, fontweight="bold", loc="left")

    fig.suptitle(
        f"Seer 124M  -  {n_params / 1e6:.0f}M params,  {total_tokens_b:.1f}B tokens,  "
        f"final val {best_val:.2f}  (${total_cost:.0f})",
        fontsize=12,
        y=1.02,
    )
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"training_curves.{ext}", bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'training_curves.png'} and .pdf")


if __name__ == "__main__":
    main()
