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
        "wandb",
        "transformers",
        "trl",
        "git+https://github.com/KellerJordan/Muon",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")  #  package
    .add_local_dir("configs", "/root/configs")
)

wandb_secret = modal.Secret.from_name("wandb")


# ckpt_best.pt -> HF format
@app.function(image=image, volumes={CKPT_DIR: ckpt_vol}, timeout=30 * 60)
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


# Training
@app.function(
    image=image,
    gpu="L4",
    volumes={CKPT_DIR: ckpt_vol},
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
def run_sft(
    base: str = "hf_seer_124m",
    out: str = "sft_out",
    limit: int = 0,
    resume: bool = False,
):
    sft.remote(base=base, out=out, limit=limit, resume=resume)
