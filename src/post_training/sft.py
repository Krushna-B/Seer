import argparse
import os
import yaml

import torch
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


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


class CombinedOptimizer(torch.optim.Optimizer):
    """Present several optimizers as one, so TRL's optimizers=(opt, None) works.

    param_groups are shared references, so HF's LR scheduler drives every group.
    """

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]
        self.defaults = {}

    @property
    def state(self):
        s = {}
        for o in self.optimizers:
            s.update(o.state)
        return s

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["opts"]):
            o.load_state_dict(s)


def build_muon_optimizer(model, ocfg, weight_decay):
    """Muon on 2D hidden matrices; AdamW on embeddings/head/norms/biases."""
    seen, hidden, adam = set(), [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and not any(k in name for k in ("wte", "wpe", "lm_head")):
            hidden.append(p)
        else:
            adam.append(p)
    muon = torch.optim.Muon(
        hidden,
        lr=float(ocfg["muon_lr"]),
        momentum=float(ocfg["muon_momentum"]),
        weight_decay=weight_decay,
    )
    adamw = torch.optim.AdamW(
        adam,
        lr=float(ocfg["embed_lr"]),
        betas=tuple(ocfg["adam_betas"]),
        eps=float(ocfg["adam_eps"]),
        weight_decay=weight_decay,
    )
    return CombinedOptimizer([muon, adamw])


def main():
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sft_alpaca.yaml")
    p.add_argument("--key", default="sft_alpaca", help="top-level config key")
    p.add_argument("--model", required=True, help="HF base model path (converted)")
    p.add_argument("--out", required=True, help="output dir for the SFT model")
    p.add_argument("--limit", type=int, default=None, help="use N examples")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint")
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
    ds = ds.train_test_split(test_size=tcfg["eval_ratio"], seed=tcfg["seed"])

    os.environ["WANDB_PROJECT"] = tcfg["wandb_project"]

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
        # Loggigng Info
        logging_first_step=True,
        report_to="wandb",
        run_name=tcfg["run_name"],
        eval_strategy="steps",  # Evaluation config and ckpt
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=tcfg["seed"],
    )

    ocfg = tcfg.get("optim", {})
    if ocfg.get("type") == "muon":
        optimizer = build_muon_optimizer(model, ocfg, float(tcfg["weight_decay"]))
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            optimizers=(optimizer, None),
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
        )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved SFT model -> {args.out}")


if __name__ == "__main__":
    main()
