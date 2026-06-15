import math
import numpy as np
import torch
import yaml
from models.GPT_model import GPT_Config, GPT_Model
from pathlib import Path
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "configs" / "seer_304m.yaml"

train_shard = (
    "/Volumes/Crucial X9/Seer/processed/fineweb-edu-10bt/seer_train_000001.bin"
)

# Load configuration yaml
with open(config_path) as f:
    cfg = yaml.safe_load(f)["seer_304m"]
    mcfg, tcfg, scfg = cfg["model"], cfg["train"], cfg["system"]

# Set seed
torch.manual_seed(scfg["seed"])

# Device & precision
device = "cuda" if torch.cuda.is_available() else scfg["device"]
torch.set_float32_matmul_precision("high")

# Build model and move model to device
model = GPT_Model(GPT_Config(**mcfg))
model.to(device)


# Optimzer
optimizer = model.configure_optimizers(
    weight_decay=tcfg["weight_decay"],
    learning_rate=tcfg["learning_rate"],
    betas=(tcfg["beta1"], tcfg["beta2"]),
    device=device,
)


def get_batch(shard_path, batch_size, block_size, device):
    """Get a sample batch of sequeces from corpus and move to device"""
    # Memmap the data path as a flat uint16
    data = np.memmap(shard_path, dtype=np.uint16, mode="r")

    # Pick batch size
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # build inputs and targets
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
            for i in ix
        ]
    )
    # move to devide
    x = x.to(device)
    y = y.to(device)
    return x, y


def get_lr(step, warmup_steps, max_steps, learning_rate, min_lr):
    # linear warmup for the first warmup_steps
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    # after max_steps, stay at min_lr
    if step > max_steps:
        return min_lr
    # in between: cosine decay from learning_rate down to min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # goes 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    grad_accum = tcfg["total_batch_size"] // (
        tcfg["micro_batch_size"] * tcfg["block_size"]
    )
    assert (
        tcfg["total_batch_size"] % (tcfg["micro_batch_size"] * tcfg["block_size"]) == 0
    )

    model.train()
    pbar = tqdm(range(tcfg["max_steps"]), desc="training")
    for step in pbar:
        lr = get_lr(
            step,
            tcfg["warmup_steps"],
            tcfg["max_steps"],
            tcfg["learning_rate"],
            tcfg["min_lr"],
        )
        for optim in optimizer.param_groups:
            optim["lr"] = lr

        optimizer.zero_grad()
        loss_accum = 0.0
        for micro in tqdm(range(grad_accum), desc=f"step {step}", leave=False):
            x, y = get_batch(
                train_shard, tcfg["micro_batch_size"], tcfg["block_size"], device
            )

            logits, loss = model(x, y)
            loss = loss / grad_accum
            loss_accum += loss.item()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()

    pbar.set_postfix(
        loss=f"{loss_accum:.3f}", lr=f"{lr:.1e}", norm=f"{float(norm):.2f}"
    )
