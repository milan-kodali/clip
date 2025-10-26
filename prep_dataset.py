"""
process & save dataset to tar shards
takes ~3 hours on a Macbook
can likely be sped-up significantly
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np

from PIL import Image
import tiktoken
from dataclasses import dataclass
from datasets import load_dataset

import os, io, tarfile

from clip import TextConfig, TextTokenizer

max_images = 2097152
img_size = 224
shard_size = 32768
n_val_shard = 1

shard_dir = "./.cache/clip_data/text-to-image-2M"
os.makedirs(shard_dir, exist_ok=True)

text_config = TextConfig()
tokenizer = TextTokenizer(text_config)
# load dataset in streaming mode
base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
num_shards = 46
urls = [base_url.format(i=i) for i in range(num_shards)]
dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)

resize = transforms.Compose([
    transforms.Resize((img_size, img_size)),
])

def process_sample(sample):
    # get image and prompt
    image = sample["jpg"]
    prompt = sample["json"]["prompt"]
    # resize image to img_size
    image = resize(image)
    # convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    # encode prompt
    tokens = tokenizer.encode(prompt)    
    # convert tokens to bytes
    token_bytes = io.BytesIO()
    torch.save(tokens, token_bytes)
    token_bytes.seek(0)
    return {"img_bytes": img_bytes, "token_bytes": token_bytes}

import time; start_time = time.time()

# process samples and create tar shards
img_index = 0
next_shard_index = 0
current_shard_count = 0
tar_file = None
tar_path = None

processed_dataset = dataset.map(process_sample)

def stream_rows():
    for row in processed_dataset:
        yield row

for sample in stream_rows():
    img_bytes = sample["img_bytes"]
    token_bytes = sample["token_bytes"]
    if img_index >= max_images:
        break
    # start new shard if needed
    if current_shard_count == 0:
        if tar_file is not None:
            tar_file.close()
            print(f"Closed shard {next_shard_index - 1}")
        split = "val" if next_shard_index < n_val_shard else "train"
        # create new tar shard
        tar_path = os.path.join(shard_dir, f"text_to_image_{split}_{next_shard_index:06d}.tar")
        tar_file = tarfile.open(tar_path, "w")
        next_shard_index += 1

    # add image to tar
    img_name = f"{img_index:07d}.jpg"
    img_info = tarfile.TarInfo(name=img_name)
    img_info.size = len(img_bytes.getvalue())
    tar_file.addfile(img_info, img_bytes)
    # add tokens to tar
    token_name = f"{img_index:07d}.pth"
    token_info = tarfile.TarInfo(name=token_name)
    token_info.size = len(token_bytes.getvalue())
    tar_file.addfile(token_info, token_bytes)
    # update indices and reset shard counter if full
    img_index += 1
    current_shard_count += 1
    if current_shard_count >= shard_size:
        current_shard_count = 0

    if img_index % 200 == 0:
        print(f"Processed {img_index}/{max_images} images")

# close final shard
if tar_file is not None:
    tar_file.close()

print(f"Done! Created {next_shard_index} shards with {img_index} images total.")

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
