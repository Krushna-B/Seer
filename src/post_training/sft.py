import argparse
import yaml

from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from muon import MuonWithAuxAdam


def to_pc(ex):
    """Map an Alpaca row to TRL's prompt/completion schema.

    TRL splits on this boundary itself: it appends EOS to the completion and
    masks the prompt tokens out of the loss.
    """
    instr = ex["instruction"] + (("\n\n" + ex["input"]) if ex["input"] else "")
    return {
        "prompt": f"### Instruction:\n{instr}\n\n### Response:\n",
        "completion": ex["output"],
    }


def build_muon_optimizer(model, ocfg, weight_decay):
    seen, head, embed, scalar, hidden = set(), [], [], [], []
    for name, p in model.parameters():
        if not p.require_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if "lm_head" in name:
            head.append(p)
        elif "wte" in name or "wpe" in name:
            embed.append(p)
        elif p.ndim >= 2:
            hidden.append(p)
        else:
            scalar.append(p)
    betas, eps = tuple(ocfg["adam_betas"]), float(ocfg["adam_eps"])
    adam = [(embed, "embed"), (head, "head_lr"), (scalar, "scalar")]
    groups = [
        dict(
            params=ps,
            lr=float(ocfg[key]),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_muon=False,
        )
        for ps, p in adam
    ]
    if hidden:
        groups.append(
            dict(
                params=hidden,
                lr=float(ocfg["muon_lr"]),
                momentum=float(ocfg["muon_momentum"]),
                weight_decay=weight_decay,
                use_muon=True,
            )
        )
    return MuonWithAuxAdam(groups)


def main():
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sft_alpaca.yaml")
    p.add_argument("--key", default="sft_alpaca", help="top-level config key")
    p.add_argument("--model", required=True, help="HF base model path (converted)")
    p.add_argument("--out", required=True, help="output dir for the SFT model")
    p.add_argument("--limit", type=int, default=None, help="use N examples")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)[args.key]
    dcfg, tcfg = cfg["data"], cfg["train"]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    split = f"train[:{args.limit}]" if args.limit else "train"
    ds = load_dataset(dcfg["dataset"], split=split).map(
        to_pc, remove_columns=["instruction", "input", "output"]
    )

    sft_cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        max_length=dcfg["max_len"],
    )

    ocfg = tcfg.get("optim", {})
    if ocfg.get("type") == "muon":
        optimizer = build_muon_optimizer(model, ocfg, float(tcfg["weight_decay"]))
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds, optimizers=(optimizer, None)
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved SFT model -> {args.out}")


if __name__ == "__main__":
    main()
