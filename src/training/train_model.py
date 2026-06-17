import math
import numpy as np
import torch
import yaml
from contextlib import nullcontext
from models.GPT_model import GPT_Config, GPT_Model
from pathlib import Path
from tqdm import tqdm
import os

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

# Mixed-precision autocast: bf16 on GPU, plain fp32 on CPU/Mac.
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[scfg["dtype"]]
ctx = (
    torch.autocast(device_type=device_type, dtype=ptdtype)
    if device_type == "cuda"
    else nullcontext()
)

# Build model and move model to device
model = GPT_Model(GPT_Config(**mcfg))
model.to(device)
raw_model = model  # uncompiled handle, for clean state_dict saves
if scfg["compile"]:
    model = torch.compile(model)


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


def save_checkpoint(path, raw_model, optimizer, step, cfg):
    """Atomic checkpoint save"""
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "cfg": cfg,
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    out_dir = Path(scfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    start_step = 0
    # If in loading state then resume from checkpoint
    if scfg["resume"] and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])  # load into raw, not compiled
        optimizer.load_state_dict(ckpt["optimizer"])  # optimizer momentum matters!
        start_step = ckpt["step"] + 1
        print(f"resumed from {ckpt_path} at step {start_step}")

    grad_accum = tcfg["total_batch_size"] // (
        tcfg["micro_batch_size"] * tcfg["block_size"]
    )
    assert (
        tcfg["total_batch_size"] % (tcfg["micro_batch_size"] * tcfg["block_size"]) == 0
    )

    model.train()
    pbar = tqdm(range(start_step, tcfg["max_steps"]), desc="training")
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

            with ctx:
                logits, loss = model(x, y)
                loss = loss / grad_accum
            loss_accum += loss.item()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss_accum:.3f}", lr=f"{lr:.1e}", norm=f"{float(norm):.2f}"
        )
        if step > 0 and step % tcfg["eval_interval"] == 0:
            save_checkpoint(ckpt_path, raw_model, optimizer, step, cfg)

    # final save — outside the loop
    save_checkpoint(ckpt_path, raw_model, optimizer, tcfg["max_steps"] - 1, cfg)
