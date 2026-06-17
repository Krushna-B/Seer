import math
import torch
import yaml
from contextlib import nullcontext
from models.GPT_model import GPT_Config, GPT_Model
from data.prepare import Shard_Loader
from pathlib import Path
from tqdm import tqdm
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "configs" / "seer_304m.yaml"

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


@torch.no_grad()
def evaluate(model, loader, ctx, eval_iters, micro_batch_size, block_size, device):
    """Mean loss over eval_iters random val batches (no grad, dropout off)."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = loader.get_batch(micro_batch_size, block_size, device)
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(path, raw_model, optimizer, step, best_val_loss, cfg):
    """Atomic checkpoint save"""
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "cfg": cfg,
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    out_dir = Path(scfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"  # latest, for resume
    best_path = out_dir / "ckpt_best.pt"  # lowest val loss

    train_loader = Shard_Loader(scfg["data_dir"], "train")
    val_loader = Shard_Loader(scfg["data_dir"], "val")

    start_step = 0
    best_val_loss = float("inf")
    # If in loading state then resume from checkpoint
    if scfg["resume"] and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])  # load into raw, not compiled
        optimizer.load_state_dict(ckpt["optimizer"])  # optimizer momentum matters!
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["best_val_loss"]
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

        # periodic val eval + checkpoints
        if step % tcfg["eval_interval"] == 0:
            val_loss = evaluate(
                model,
                val_loader,
                ctx,
                tcfg["eval_iters"],
                tcfg["micro_batch_size"],
                tcfg["block_size"],
                device,
            )
            tqdm.write(
                f"step {step}: val_loss {val_loss:.4f} (best {best_val_loss:.4f})"
            )
            save_checkpoint(ckpt_path, raw_model, optimizer, step, best_val_loss, cfg)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_path, raw_model, optimizer, step, best_val_loss, cfg
                )

        optimizer.zero_grad()
        loss_accum = 0.0
        for micro in tqdm(range(grad_accum), desc=f"step {step}", leave=False):
            x, y = train_loader.get_batch(
                tcfg["micro_batch_size"], tcfg["block_size"], device
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

    # final save — outside the loop
    save_checkpoint(
        ckpt_path, raw_model, optimizer, tcfg["max_steps"] - 1, best_val_loss, cfg
    )
