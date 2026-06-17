import json
import time
import platform
import subprocess
from pathlib import Path
import torch


def _git_sha():
    """Return full sha hash of current repo commit"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_card(cfg, n_params, device, seed, data_dir, out_dir):
    """ "All info at start of run"""
    cuda = torch.cuda.is_available()
    return {
        "status": "running",
        "git_sha": _git_sha(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "cfg": cfg,
        "data_dir": data_dir,
        "out_dir": out_dir,
        "n_params": n_params,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
            "gpu_mem_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
            if cuda
            else None,
        },
    }


def write_card(path, card):
    Path(path).write_text(json.dumps(card, indent=2, default=str))
