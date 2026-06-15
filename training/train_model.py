import numpy as np
import torch


def get_batch(shard_path, batch_size, block_size, device):
    """Get a sample batch of sequeces from corpus and move to device"""
    # Memmap the data path as a flat uint16
    data = np.memmap(shard_path, dtype=np.uint16, mode="r")

    # Pick batch size
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # build inputs and targets
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
            for i in ix
        ]
    )
    # move to devide
    x = x.to(device)
    y = y.to(device)
    return x, y
