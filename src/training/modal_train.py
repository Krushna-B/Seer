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
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")  #  package
    .add_local_dir("configs", "/root/configs")
)
# Persistent Storage
data_vol = modal.Volume.from_name("seer-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("seer-ckpt", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
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


@app.local_entrypoint()
def main(resume: bool = False):
    train.remote(resume=resume)
