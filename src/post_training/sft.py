import argparse
import yaml

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

    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved SFT model -> {args.out}")


if __name__ == "__main__":
    main()
