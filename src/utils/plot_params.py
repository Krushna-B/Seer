"""Parameter breakdown of the Seer 124M architecture.

Computed analytically from the config (verified against the model's real
124,439,808 count). Shows where the parameters actually live — notably that
~31% of a GPT-2-small is the token-embedding table, and the head is weight-tied
to it (so it adds nothing).
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

# palette: MLP / attention emphasized, embeddings muted, tiny norms faint
COLORS = {
    "MLP (12 blocks)": "#1f4e79",
    "Attention (12 blocks)": "#3d6fa5",
    "Token embeddings": "#c1432e",
    "Position embeddings": "#e0a08f",
    "LayerNorms": "#b4b2a9",
}


def breakdown(cfg):
    V, C, L, d = cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_embd"]
    b = 1 if cfg.get("bias", True) else 0
    attn = L * ((d * 3 * d + 3 * d * b) + (d * d + d * b))  # c_attn + oproj
    mlp = L * ((d * 4 * d + 4 * d * b) + (4 * d * d + d * b))  # fc + proj
    norms = L * 2 * (d + d * b) + (d + d * b)  # 2 per block + final
    return {
        "Token embeddings": V * d,
        "Position embeddings": C * d,
        "Attention (12 blocks)": attn,
        "MLP (12 blocks)": mlp,
        "LayerNorms": norms,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", default=str(ROOT / "artifacts/hf_seer_124m/config.json")
    )
    p.add_argument("--out", default=str(ROOT / "results/figures/param_breakdown"))
    args = p.parse_args()

    hf = json.load(open(args.config))
    cfg = {
        "vocab_size": hf["vocab_size"],
        "block_size": hf.get("n_positions", hf.get("n_ctx", 1024)),
        "n_layer": hf["n_layer"],
        "n_embd": hf["n_embd"],
        "bias": True,
    }
    parts = breakdown(cfg)
    total = sum(parts.values())

    order = [
        "MLP (12 blocks)",
        "Attention (12 blocks)",
        "Token embeddings",
        "Position embeddings",
        "LayerNorms",
    ]
    labels = order
    vals = [parts[k] for k in order]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = range(len(labels))
    bars = ax.barh(
        list(y), [v / 1e6 for v in vals], color=[COLORS[k] for k in labels]
    )
    ax.invert_yaxis()
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("parameters (millions)")
    ax.set_xlim(0, max(vals) / 1e6 * 1.18)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_width() + total / 1e6 * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{v / 1e6:.1f}M  ({v / total * 100:.1f}%)",
            va="center",
            fontsize=9.5,
            color="#2c2c2a",
        )

    ax.set_title(
        f"Seer 124M — parameter breakdown  ({total:,} total, head weight-tied)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.png and .pdf   (total={total:,})")


if __name__ == "__main__":
    main()
