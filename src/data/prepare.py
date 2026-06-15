import tiktoken
import numpy as np
import os
from tqdm import tqdm
from download import download_data


# GPT2 Encoder with vocab size of 50.257
ENCODING = tiktoken.get_encoding("gpt2")
EOT = ENCODING._special_tokens["<|endoftext|>"]

OUPUT_DIR = "/Volumes/Crucial X9/Seer/processed/fineweb-edu-10bt"
SHARD_SIZE = int(1e8)
# sample-10BT is ~10B tokens; only used to drive the overall progress bar / ETA
TOTAL_TOKENS_EST = int(1e10)


def tokenize(doc: dict):
    """Turn one document into a uint16 token array with EOT prepended"""
    tokens = [EOT]
    tokens.extend(ENCODING.encode_ordinary(doc["text"]))
    tokens = np.array(tokens, dtype=np.uint16)
    assert (tokens < 2**16).all()
    return tokens


def shard_tokens(data):
    os.makedirs(OUPUT_DIR, exist_ok=True)
    buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    count = 0
    shard_index = 0
    expected_shards = TOTAL_TOKENS_EST // SHARD_SIZE
    # one overall bar across the whole run: gives % done, tok/s rate, and ETA
    progress = tqdm(
        total=TOTAL_TOKENS_EST, unit="tok", unit_scale=True, desc="tokenizing"
    )
    progress.set_postfix(shard=0, est_total=expected_shards)

    for doc in data:
        # Token length is less than sharding bucket
        tokens = tokenize(doc=doc)
        progress.update(len(tokens))
        if len(tokens) + count < SHARD_SIZE:
            buf[count : count + len(tokens)] = tokens
            count += len(tokens)
        else:
            # Token length is more than sharding bucket
            remainder = SHARD_SIZE - count
            buf[count : count + remainder] = tokens[:remainder]
            split = "val" if shard_index == 0 else "train"
            path = os.path.join(OUPUT_DIR, f"seer_{split}_{shard_index:06d}.bin")
            # Write to bin output
            buf.tofile(path)
            shard_index += 1
            progress.set_postfix(shard=shard_index, est_total=expected_shards)
            # if shard_index % 5 == 0:
            # # tqdm.write keeps the bar intact instead of print() shredding it
            # tqdm.write(
            #     f"wrote {shard_index}/{expected_shards} shards ({shard_index * SHARD_SIZE * 2 / 1000**2:.0f} MB) in {OUPUT_DIR}"
            # )

            leftover = len(tokens) - remainder
            buf[0:leftover] = tokens[remainder:]
            count = leftover

    progress.close()
    # final shard
    if count > 0:
        split = "val" if shard_index == 0 else "train"
        path = os.path.join(OUPUT_DIR, f"seer_{split}_{shard_index:06d}.bin")
        buf[:count].tofile(path)
    print(
        f"finished: wrote {shard_index + 1} shards in total. Total size {((shard_index + 1) * SHARD_SIZE * np.dtype(np.uint16).itemsize) / 1000**2} MB to {OUPUT_DIR}"
    )


if __name__ == "__main__":
    fw = download_data("sample-10BT")
    fw = fw.select_columns(["text"])
    shard_tokens((fw))
