"""Hellaswag eval for pretrained model"""

import argparse
import json
import os
import urllib.request

import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import pick_device, load_model

ENCODING = tiktoken.get_encoding("gpt2")
DATA_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
)


def download_val(cache_dir):
    """Pull the raw jsonl"""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "hellaswag_val.jsonl")
    if not os.path.exists(path):
        print(f"downloading hellaswag val -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def render_example(example, device):
    """Turn example into (4,T) batch for every encoding"""
    """ row is the inital context tokens + ending token, context is masked out,
        and rows of different lengths are padded out"""
    ctx_tokens = ENCODING.encode(example["ctx"])
    rows, masks = [], []
    for ending in example["endings"]:
        end_tokens = ENCODING.encode(" " + ending)
        rows.append(ctx_tokens + end_tokens)
        masks.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    maxlen = max(len(r) for r in rows)
    tokens = torch.zeros(4, maxlen, dtype=torch.long)
    mask = torch.zeros(4, maxlen, dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tokens[i, : len(r)] = torch.tensor(r)
        mask[i, : len(m)] = torch.tensor(m)
    return tokens.to(device), mask.to(device), int(example["label"])


@torch.no_grad
def score_example(model, tokens, mask):

    logits, _ = model(tokens, targets=tokens)
    pred = logits[:, :-1, :]
    gold = tokens[:, 1:]
    vocab = pred.size(-1)

    losses = F.cross_entropy(
        pred.reshape(-1, vocab), gold.reshape(-1), reduction="none"
    ).view(tokens.size(0), -1)  # (4, T-1) per-token loss

    shift_mask = mask[:, 1:].float()
    masked = losses * shift_mask
    sum_loss = masked.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return sum_loss.argmin().item(), avg_loss.argmin().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/ckpt_best.pt")
    p.add_argument("--cache_dir", default="data/eval")
    p.add_argument("--limit", type=int, default=10, help="eval only N examples")
    args = p.parse_args()

    device = pick_device()
    model, *_ = load_model(args.ckpt, device)
    path = download_val(args.cache_dir)

    n_correct = n_correct_norm = n_total = 0
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    if args.limit is not None:
        examples = examples[: args.limit]

    for ex in tqdm(examples, desc="Hellaswag"):
        tokens, mask, label = render_example(ex, device)
        pred, pred_norm = score_example(model, tokens, mask)
        n_correct += int(pred == label)
        n_correct_norm += int(pred_norm == label)
        n_total += 1
    print(f"\nn={n_total}")
    print(f"acc      {n_correct / n_total:.4f}")
    print(f"acc_norm {n_correct_norm / n_total:.4f}")


if __name__ == "__main__":
    main()
