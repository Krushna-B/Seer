import argparse
import json
import yaml
import os

from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig
from datasets import load_dataset


def preprocess_function(example):
    """Map template from alpaca instruction to DPOtrainer example"""
    instruction = example["chosen"][0]["content"]
    return {
        "prompt": f"### Instruction:\n{instruction}\n\n### Response:\n",
        "chosen": example["chosen"][-1]["content"],
        "rejected": example["rejected"][-1]["content"],
    }


def main():
    # Inputs args
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        required=True,
        default="artifacts/sft_out/checkpoint-1500",
    )
    p.add_argument("--config", default="configs/seer_304m.yaml")
    p.add_argument("--key", default="dpo")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="use N train examples")

    # Load configurations
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)[args.key]
    dcfg, tcfg = cfg["data"], cfg["train"]
    os.environ["WANDB_PROJECT"] = tcfg["wandb_project"]
    dpo_cfg = DPOConfig(
        output_dir=args.out,
        beta=tcfg["beta"],
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=float(tcfg["lr"]),
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

    # Load Data
    ds = load_dataset(path=dcfg["data"])
    if args.limit:
        ds["train"] = ds["train"].select(range(args.limit))
    ds = ds.map(preprocess_function, remove_columns=ds["train"].column_names)

    # Run Hugging face DPOTrainer
    trainer = DPOTrainer(
        model=args.model,
        args=dpo_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)

    # Store metric history
    log_path = os.path.join(args.out, "dpo_log.jsonl")
    with open(log_path, "w") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row) + "\n")
    print(f"saved dpo model -> {args.out}  (log -> {log_path})")


if __name__ == "__main__":
    main()
