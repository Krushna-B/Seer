import argparse

import tiktoken
import torch
import torch.nn.functional as F

from models.GPT_model import GPT_Config, GPT_Model

ENCODING = tiktoken.get_encoding("gpt2")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    """Rebuild the model from the checkpoint's saved config and weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt["cfg"]["model"]
    model = GPT_Model(GPT_Config(**mcfg))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model, mcfg["block_size"], ckpt.get("step"), ckpt.get("best_val_loss")


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature, top_k):
    """Autoregressive sampling. No KV cache."""
    for _ in range(max_new_tokens):
        # crop context to the model's block size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)  # (B, 1, vocab) when no targets
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/ckpt_best.pt")
    p.add_argument("--prompt", default="The mitochondria is")
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    model, block_size, step, val_loss = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} | step {step} | val_loss {val_loss} | device {device}\n")

    start = ENCODING.encode_ordinary(args.prompt)
    idx = torch.tensor(start, dtype=torch.long, device=device)[None, :]

    for i in range(args.num_samples):
        out = generate(
            model, idx, args.tokens, block_size, args.temperature, args.top_k
        )
        text = ENCODING.decode(out[0].tolist())
        print(f"--- sample {i + 1} ---\n{text}\n")


if __name__ == "__main__":
    main()
