"""
simple data loader for text-to-image-2M dataset
loads each shard tarfile into memory for faster access
reasonable performance for up to B=4096
"""

import os
import io
import tarfile
from PIL import Image
import torch
from torchvision import transforms

# image to tensor
itot = transforms.ToTensor()

def load_shard(shard_path, verbose = False):
    data = {}
    with tarfile.open(shard_path, "r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                data[member.name] = f.read()
    if verbose: print(f"loaded {len(data)} files ({len(data) // 2} image-text pairs) from {shard_path.split('/')[-1]}\n-----")
    return data

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
        self.data = load_shard(self.shards[self.shard_index])
        self.files = sorted(self.data.keys())
        self.current_position = self.rank * (self.B * 2)
    
    def get_state(self):
        return {
            'shard_index': self.shard_index,
            'current_position': self.current_position,
        }
    
    def load_state(self, state):
        self.shard_index = state['shard_index']
        self.current_position = state['current_position']
        self.data = load_shard(self.shards[self.shard_index], verbose=self.verbose)
        self.files = sorted(self.data.keys())
    
    def next_batch(self):
        B = self.B
        images, tokens = [], []
        for i in range(B):
            img_name = self.files[self.current_position + i*2]
            token_name = self.files[self.current_position + i*2 + 1]
            img = Image.open(io.BytesIO(self.data[img_name]))
            images.append(itot(img))
            tokens.append(torch.load(io.BytesIO(self.data[token_name]), map_location='cpu'))
        labels = torch.stack(tokens)
        images = torch.stack(images)
        self.step(1)
        return labels, images

    def next_shard(self):
        self.shard_index = (self.shard_index + 1) % self.shard_count
        self.data = load_shard(self.shards[self.shard_index], verbose = self.verbose)
        self.files = sorted(self.data.keys())
        self.current_position = self.rank * (self.B * 2)

    def step(self, num_steps = 1):
        for i in range(num_steps):
            B, world_size, rank = self.B, self.world_size, self.rank
            self.current_position += (2 * B) * world_size
            # reset if next batch would be out of bounds
            next_position = self.current_position + (2 * B) * world_size
            if next_position > len(self.files):
                self.next_shard()
                self.current_position = rank * (2 * B)

if __name__ == "__main__":
    import time
    from clip import TextTokenizer, TextConfig
    enc = TextTokenizer(TextConfig())
    B = 4096
    torch.manual_seed(42)
    print(f"loading & timing a preview batch with B = {B}\n-----")
    t0 = time.time()
    train_loader = DataLoaderLite(B = B, rank = 0, world_size = 1, split = 'train', verbose = True)
    t1 = time.time()
    print(f"⏱️ initializing DataLoaderLite took {t1 - t0:.2f}s\n-----")
    # timed batches
    batches = 10
    for i in range(batches):
        t0 = time.time()
        labels, images = train_loader.next_batch() 
        t1 = time.time()
        print(f"⏱️ loading batch took {t1 - t0:.3f}s")
        time.sleep(2) # simulate batch processing time
        # preview last batch
        if i == batches - 1:
            print(f"> last batch shape: {images.shape}, {labels.shape}")
            print(f"> last batch label preview: {enc.decode(labels[325])}\n")