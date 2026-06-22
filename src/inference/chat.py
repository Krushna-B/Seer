import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.utils import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/sft_out/checkpoint-1500")
    p.add_argument(
        "--prompt", default="Explain what a mitochondrion does in one sentence."
    )
    p.add_argument("--input", default="")
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    instr = args.prompt + (("\n\n" + args.input) if args.input else "")
    text = f"### Instruction:\n{instr}\n\n### Response:\n"  # the trained format
    ids = tok(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=args.tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=True,  # KV cache
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    print(tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
