"""
simple data loader for text-to-image-2M dataset
loads each shard tarfile into memory for faster access
reasonable performance for up to B=4096

2: experiments to improve perf
"""

import os
import io
import tarfile
from PIL import Image
import torch
from torchvision import transforms
import time
import webdataset as wds

# image to tensor
itot = transforms.ToTensor()

from concurrent.futures import ThreadPoolExecutor
import io
from PIL import Image
import torch

def decode_pair(img_bytes, token_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return itot(img), torch.load(io.BytesIO(token_bytes), map_location='cpu')

def load_shard(shard_path, num_threads=8, verbose=False):
    # Step 1: Read all files into memory (raw bytes)
    with tarfile.open(shard_path, "r") as tar:
        files = sorted([m for m in tar.getmembers() if m.isfile()], key=lambda m: m.name)
        names = [m.name for m in files]
        blobs = [tar.extractfile(m).read() for m in files]

    # Step 2: Pair image/text files sequentially
    pairs = [(blobs[i], blobs[i+1]) for i in range(0, len(blobs), 2)]

    # Step 3: Parallel decode
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        decoded = list(ex.map(lambda p: decode_pair(*p), pairs))

    # Step 4: Split into two lists for easy batching
    images, tokens = zip(*decoded)
    return list(images), list(tokens)

class DataLoaderLite:
    def __init__(self, B, rank, world_size, split = 'train', verbose = True):
        self.B = B
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose

        assert split in ['train', 'val'], "split must be either 'train' or 'val'"
        self.split = split

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'clip_data', 'text-to-image-2M')
        shards = [f for f in os.listdir(data_dir) if f.startswith(f'text_to_image_{self.split}') and f.endswith('.tar')]
        shards = [os.path.join(data_dir, f) for f in shards]
        self.shards = sorted(shards)
        self.shard_count = len(self.shards)
        assert self.shard_count > 0, f"no shards found for {split} split"
        if self.verbose: print(f"found {len(self.shards)} shards for {split} split")

        # set initial state
        self.reset()
        
    def reset(self):
        self.shard_index = 0
        self.images, self.labels = load_shard(self.shards[self.shard_index])
        self.current_position = self.rank * (self.B * 2)
    
    def get_state(self):
        return {
            'shard_index': self.shard_index,
            'current_position': self.current_position,
        }
    
    def load_state(self, state):
        self.shard_index = state['shard_index']
        self.current_position = state['current_position']
        self.images, self.labels = load_shard(self.shards[self.shard_index])
    
    def next_batch(self):
        start = self.current_position
        end = start + self.B
        images = torch.stack(self.images[start:end])
        labels = torch.stack(self.labels[start:end])
        self.step(1)
        return labels, images

    def next_shard(self):
        self.shard_index = (self.shard_index + 1) % self.shard_count
        self.images, self.labels = load_shard(self.shards[self.shard_index])
        self.current_position = self.rank * (self.B * 2)

    def step(self, num_steps = 1):
        for i in range(num_steps):
            B, world_size, rank = self.B, self.world_size, self.rank
            self.current_position += (2 * B) * world_size
            # reset if next batch would be out of bounds
            next_position = self.current_position + (2 * B) * world_size
            if next_position > len(self.images):
                self.next_shard()
                self.current_position = rank * (2 * B)

if __name__ == "__main__":
    import time
    from clip import TextTokenizer, TextConfig
    enc = TextTokenizer(TextConfig())
    B = 4096
    torch.manual_seed(42)
    print(f"loading & timing a preview batch with B = {B}\n-----")
    start_time = time.time()
    train_loader = DataLoaderLite(B, 0, 1, split = 'train', verbose = True)
    end_time = time.time()
    print(f"⏱️ initializing DataLoaderLite took {end_time - start_time:.2f}s\n-----")
    # discard warmup batches
    labels, images = train_loader.next_batch()    

    
    # timed batch
    start_time = time.time()
    labels, images = train_loader.next_batch()
    print(f"> batch shape: {images.shape}, {labels.shape}")
    print(f"> preview of label: {enc.decode(labels[0])}\n")
    end_time = time.time()
    print(f"⏱️ batch took {end_time - start_time:.2f}s")

# MacBook logs with B=4096:
# baseline: 10s init, 1s batch