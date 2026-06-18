import argparse

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from models.GPT_model import GPT_Config, GPT_Model

# our Linear weights -> HF Conv1D weights need transposing
TRANSPOSE = {
    "attn.attn.weight",
    "attn.oproj.weight",
    "mlp.fc_layer.weight",
    "mlp.proj_layer.weight",
}

# per-block name map: ours -> HF
BLOCK_MAP = {
    "ln_1.weight": "ln_1.weight",
    "ln_1.bias": "ln_1.bias",
    "attn.attn.weight": "attn.c_attn.weight",
    "attn.attn.bias": "attn.c_attn.bias",
    "attn.oproj.weight": "attn.c_proj.weight",
    "attn.oproj.bias": "attn.c_proj.bias",
    "ln_2.weight": "ln_2.weight",
    "ln_2.bias": "ln_2.bias",
    "mlp.fc_layer.weight": "mlp.c_fc.weight",
    "mlp.fc_layer.bias": "mlp.c_fc.bias",
    "mlp.proj_layer.weight": "mlp.c_proj.weight",
    "mlp.proj_layer.bias": "mlp.c_proj.bias",
}


def build_hf_config(mcfg):
    return GPT2Config(
        vocab_size=mcfg["vocab_size"],
        n_positions=mcfg["block_size"],
        n_ctx=mcfg["block_size"],
        n_embd=mcfg["n_embd"],
        n_layer=mcfg["n_layer"],
        n_head=mcfg["n_head"],
        activation_function="gelu",  # GOTCHA 2: match nn.GELU (exact), not gelu_new
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,  # matches nn.LayerNorm default
        bos_token_id=50256,
        eos_token_id=50256,
    )


def convert_state(sd, n_layer):
    out = {
        "transformer.wte.weight": sd["transformer.wte.weight"],
        "transformer.wpe.weight": sd["transformer.wpe.weight"],
        "transformer.ln_f.weight": sd["transformer.norm_layer.weight"],
        "transformer.ln_f.bias": sd["transformer.norm_layer.bias"],
        "lm_head.weight": sd["transformer.wte.weight"],  # tied
    }
    for i in range(n_layer):
        for ours, theirs in BLOCK_MAP.items():
            w = sd[f"transformer.heads.{i}.{ours}"]
            if ours in TRANSPOSE:  # GOTCHA 1: Conv1D = transposed Linear
                w = w.t().contiguous()
            out[f"transformer.h.{i}.{theirs}"] = w
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/ckpt_best.pt")
    p.add_argument("--out", default="artifacts/hf_seer_124m")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt["cfg"]["model"]

    # rebuild ours (reference for the round-trip check)
    ours = GPT_Model(GPT_Config(**mcfg)).eval()
    ours.load_state_dict(ckpt["model"])

    # build HF and load converted weights
    hf = GPT2LMHeadModel(build_hf_config(mcfg)).eval()
    missing, unexpected = hf.load_state_dict(
        convert_state(ckpt["model"], mcfg["n_layer"]), strict=False
    )
    # only HF's non-persistent causal-mask buffers should be "missing"
    missing = [
        m for m in missing if not m.endswith((".attn.bias", ".attn.masked_bias"))
    ]
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"

    # round-trip: logits must match
    ids = torch.randint(0, mcfg["vocab_size"], (2, 64))
    ours_logits, _ = ours(ids, targets=ids)  # targets -> full logits
    hf_logits = hf(ids).logits
    max_diff = (ours_logits - hf_logits).abs().max().item()
    print(f"max logit diff: {max_diff:.2e}")
    assert max_diff < 1e-3, "conversion mismatch — do not trust eval"

    hf.save_pretrained(args.out)
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(args.out)
    print(f"saved HF model + tokenizer -> {args.out}")


if __name__ == "__main__":
    main()
