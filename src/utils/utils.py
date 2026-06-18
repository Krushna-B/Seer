import torch
from models.GPT_model import GPT_Config, GPT_Model


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    """Rebuild the model from the checkpoint's saved config and weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt["cfg"]["model"]
    model = GPT_Model(GPT_Config(**mcfg))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model, mcfg["block_size"], ckpt.get("step"), ckpt.get("best_val_loss")
