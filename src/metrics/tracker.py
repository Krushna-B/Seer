import json
import math
import time
import torch

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML = None


def _flops_per_token(n_params, n_layer, n_head, n_embd, block_size):
    head_dim = n_embd // n_head
    return 6 * n_params + 12 * n_layer * n_head * head_dim * block_size


def gpu_stats():
    if _NVML is None:
        return {}
    return {
        "gpu_util": pynvml.nvmlDeviceGetUtilizationRates(_NVML).gpu,
        "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(_NVML) / 1000,
        "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(
            _NVML, pynvml.NVML_TEMPERATURE_GPU
        ),
    }


class MetricTracker:
    def __init__(
        self,
        log_path,
        mcfg,
        n_params,
        tokens_per_step,
        price_per_hour,
        peak_flops,
        use_wandb=False,
    ):
        self.log_file = open(log_path, "a")
        self.fpt = _flops_per_token(
            n_params,
            mcfg["n_layer"],
            mcfg["n_head"],
            mcfg["n_embd"],
            mcfg["block_size"],
        )
        self.tokens_per_step = tokens_per_step
        self.price_per_hour = price_per_hour
        self.peak_flops = peak_flops
        self.use_wandb = use_wandb
        self.t_start = time.time()
        self.total_tokens = 0

    def log_step(self, step, loss, lr, grad_norm, dt, val_loss=None):
        self.total_tokens += self.tokens_per_step
        rec = {
            "step": step,
            "loss": loss,
            "ppl": math.exp(loss) if loss < 30 else float("inf"),
            "lr": lr,
            "grad_norm": grad_norm,
            "dt": dt,
            "tok_per_sec": self.tokens_per_step / dt,
            "mfu": (self.fpt * self.tokens_per_step / dt) / self.peak_flops,
            "tokens": self.total_tokens,
            "elapsed_hr": (time.time() - self.t_start) / 3600,
            "cost_usd": (time.time() - self.t_start) / 3600 * self.price_per_hour,
        }
        if torch.cuda.is_available():
            rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rec.update(gpu_stats())
        if val_loss is not None:
            rec["val_loss"] = val_loss
            rec["val_ppl"] = math.exp(val_loss)

        self.log_file.write(json.dumps(rec) + "\n")
        self.log_file.flush()
        if self.use_wandb:
            import wandb

            wandb.log(rec, step=step)
        return rec

    def close(self):
        self.log_file.close()
