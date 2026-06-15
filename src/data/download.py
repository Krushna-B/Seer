from datasets import load_dataset
from dotenv import load_dotenv
import os


def download_data(name):
    load_dotenv()
    # Load corpus from Hugging Face
    """10B Token Sample from the fine-web dataset"""
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=name,
        split="train",
        streaming=True,
        token=os.getenv("HF_TOKEN"),
    )
    return fw
