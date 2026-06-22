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
