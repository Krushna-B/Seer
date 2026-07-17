import os
import subprocess
import modal


app = modal.App("seer-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "tiktoken",
        "pyyaml",
        "tqdm",
        "huggingface_hub",
        "pynvml",
        "datasets",
        "python-dotenv",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")  #  package
    .add_local_dir("configs", "/root/configs")
)

# SFT image
sft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.12",  # torch.optim.Muon is built in from 2.12
        "numpy",
        "pyyaml",
        "datasets",
        "huggingface_hub",
        "transformers",
        "trl",
        "wandb",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("configs", "/root/configs")
)
# Persistent Storage
data_vol = modal.Volume.from_name("seer-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("seer-ckpt", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb")
DATA_DIR, CKPT_DIR = "/data", "/ckpt"


# Pull shards from hf
@app.function(
    image=image, volumes={DATA_DIR: data_vol}, secrets=[hf_secret], timeout=60 * 60
)
def download_data(repo_id: str):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns=["*.bin"],
        token=os.environ["HF_TOKEN"],
    )
    data_vol.commit()
    print("shards:", [f for f in os.listdir(DATA_DIR) if f.endswith(".bin")])


# Training job
@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol},
    timeout=24 * 60 * 60,
)
def train(resume: bool = False):
    env = {
        **os.environ,
        "SEER_DATA_DIR": DATA_DIR,
        "SEER_OUT_DIR": CKPT_DIR,
        "SEER_RESUME": "true" if resume else "false",
    }
    subprocess.run(
        ["python", "-m", "training.train_model"], cwd="/root", env=env, check=True
    )
    ckpt_vol.commit()  # persist checkpoints for next run


# ckpt_best.pt -> HF format
@app.function(image=sft_image, volumes={CKPT_DIR: ckpt_vol}, timeout=30 * 60)
def convert(ckpt: str = "ckpt_best.pt", out: str = "hf_seer_124m"):
    subprocess.run(
        [
            "python",
            "-m",
            "eval.convert_to_hf",
            "--ckpt",
            f"{CKPT_DIR}/{ckpt}",
            "--out",
            f"{CKPT_DIR}/{out}",
        ],
        cwd="/root",
        check=True,
    )
    ckpt_vol.commit()


# SFT job
@app.function(
    image=sft_image,
    gpu="L4",
    volumes={CKPT_DIR: ckpt_vol},
    secrets=[hf_secret, wandb_secret],
    timeout=3 * 60 * 60,
)
def sft(
    base: str = "hf_seer_124m",
    out: str = "sft_out",
    limit: int = 0,
    resume: bool = False,
):
    cmd = [
        "python",
        "-m",
        "post_training.sft",
        "--model",
        f"{CKPT_DIR}/{base}",
        "--out",
        f"{CKPT_DIR}/{out}",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    if resume:
        cmd += ["--resume"]
    subprocess.run(cmd, cwd="/root", env={**os.environ}, check=True)
    ckpt_vol.commit()


@app.local_entrypoint()
def main(resume: bool = False):
    train.remote(resume=resume)


@app.local_entrypoint()
def run_sft(
    base: str = "hf_seer_124m",
    out: str = "sft_out",
    limit: int = 0,
    resume: bool = False,
):
    sft.remote(base=base, out=out, limit=limit, resume=resume)


# DPO image
dpo_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.12",
        "numpy",
        "pyyaml",
        "datasets",
        "huggingface_hub",
        "transformers",
        "trl",
        "wandb",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("configs", "/root/configs")
)


# DPO Job
@app.function(
    image=dpo_image,
    gpu="L4",
    volumes={CKPT_DIR: ckpt_vol},
    secrets=[hf_secret, wandb_secret],
    timeout=2 * 60 * 60,
)
def dpo(sft_model="sft_out/checkpoint-1500", out="dpo_results", limit=0, resume=False):
    cmd = [
        "python",
        "-m",
        "post_training.dpo",
        "--model",
        f"{CKPT_DIR}/{sft_model}",
        "--out",
        f"{CKPT_DIR}/{out}",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    if resume:
        cmd += ["--resume"]
    subprocess.run(cmd, cwd="/root", env={**os.environ}, check=True)
    ckpt_vol.commit()


# Eval image
eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.12",
        "numpy",
        "transformers",
        "accelerate",
        "datasets",
        "huggingface_hub",
        "lm-eval[ifeval]",
    )
    .env({"PYTHONPATH": "/root/src", "HF_ALLOW_CODE_EVAL": "0"})
    .add_local_dir("src", "/root/src")
)

# checkpoint name -> path on the ckpt volume
EVAL_MODELS = {
    "base": "hf_seer_124m",
    "sft": "sft_out/checkpoint-1500",
    "dpo": "dpo_results",
}
CAP_TASKS = (
    "hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai,truthfulqa_mc2"
)
IF_TASKS = "ifeval"


# Eval job: lm-eval sweep over base / sft / dpo with raw per-sample logging
@app.function(
    image=eval_image,
    gpu="L4",
    volumes={CKPT_DIR: ckpt_vol},
    secrets=[hf_secret],
    timeout=6 * 60 * 60,
)
def evals(models: str = "base,sft,dpo", out: str = "evals", limit: int = 1000):
    for name in models.split(","):
        path = f"{CKPT_DIR}/{EVAL_MODELS[name]}"
        for group, tasks in (("capability", CAP_TASKS), ("ifeval", IF_TASKS)):
            cmd = [
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={path},dtype=bfloat16",
                "--tasks",
                tasks,
                "--batch_size",
                "auto",
                "--device",
                "cuda:0",
                "--output_path",
                f"{CKPT_DIR}/{out}/{name}/{group}",
                "--log_samples",
            ]
            if group == "ifeval":
                # model ctx is 1024; ifeval defaults to 1280 gen tokens -> cap it.
                # instruction-following is the headline delta -> run the full set.
                cmd += ["--gen_kwargs", "max_gen_toks=512"]
            elif limit:
                # cap the big capability tasks (hellaswag/lambada); the rest are
                # already < limit so they run complete.
                cmd += ["--limit", str(limit)]
            subprocess.run(cmd, cwd="/root", check=True)
        ckpt_vol.commit()  # persist after each checkpoint finishes


@app.local_entrypoint()
def run_evals(models: str = "base,sft,dpo", out: str = "evals", limit: int = 1000):
    evals.remote(models=models, out=out, limit=limit)


# Jinja template reproducing the exact SFT/DPO training format:
#   "### Instruction:\n{instr}\n\n### Response:\n"
ALPACA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "### Instruction:\n{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "### Response:\n{{ message['content'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}### Response:\n{% endif %}"
)


# Flexible eval: arbitrary tasks/limit, optionally apply the Alpaca chat template
# (attaches it to the tokenizer so the post-trained models are prompted in the
# format they were trained on).
@app.function(
    image=eval_image,
    gpu="L4",
    volumes={CKPT_DIR: ckpt_vol},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
)
def evals2(
    models: str,
    tasks: str,
    out: str,
    limit: int = 0,
    apply_template: bool = False,
):
    from transformers import AutoTokenizer

    for name in models.split(","):
        path = f"{CKPT_DIR}/{EVAL_MODELS[name]}"
        if apply_template:
            tok = AutoTokenizer.from_pretrained(path)
            tok.chat_template = ALPACA_CHAT_TEMPLATE
            tok.save_pretrained(path)
        cmd = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={path},dtype=bfloat16",
            "--tasks",
            tasks,
            "--batch_size",
            "auto",
            "--device",
            "cuda:0",
            "--output_path",
            f"{CKPT_DIR}/{out}/{name}",
            "--log_samples",
        ]
        if apply_template:
            cmd += ["--apply_chat_template"]
        if "ifeval" in tasks:
            cmd += ["--gen_kwargs", "max_gen_toks=512"]
        if limit:
            cmd += ["--limit", str(limit)]
        subprocess.run(cmd, cwd="/root", check=True)
        ckpt_vol.commit()


@app.local_entrypoint()
def run_evals2(
    models: str = "sft,dpo",
    tasks: str = "ifeval",
    out: str = "evals_chat",
    limit: int = 0,
    apply_template: bool = False,
):
    evals2.remote(
        models=models, tasks=tasks, out=out, limit=limit, apply_template=apply_template
    )


@app.local_entrypoint()
def run_dpo(
    base="sft_out/checkpoint-1500",
    out: str = "dpo_results",
    limit: int = 0,
    resume: bool = False,
):
    dpo.remote(sft_model=base, out=out, limit=limit, resume=resume)
