"""
simple data loader for text-to-image-2M dataset
loads each shard tarfile into memory for faster access to batches
prefetches next shard after loading the current shard

when using, need to tune shard size based on batch size &
shard-load time to minimize blocking
"""

import os
import io
import tarfile
from PIL import Image
import torch
from torchvision import transforms
import time
from concurrent.futures import ThreadPoolExecutor

# image to tensor
itot = transforms.ToTensor()
# number of decoding threads to use
n_proc = min(os.cpu_count() // 2, 32)

def decode_and_write(idx, img_bytes, token_bytes, images_tensor, labels_tensor):
    """
    parallelizable helper function to decode a pair and write directly to
    pre-allocated tensors (to reduce memory allocation overhead) at index idx
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    images_tensor[idx] = itot(img)
    labels_tensor[idx] = torch.load(io.BytesIO(token_bytes), map_location='cpu')

class DataLoader:
    def __init__(self, B = 1, block_size = 77, img_size = 224, rank = 0, world_size = 1, split = 'train', verbose = True):
        self.B = B
        self.block_size = block_size
        self.img_size = img_size
        self.rank = rank
        self.world_size = world_size
        self.n_proc = n_proc
        self.verbose = verbose
        self.images = None
        self.labels = None
        # prefetch buffers
        self.prefetch_images = None
        self.prefetch_labels = None
        self.prefetch_future = None
        self.prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch = True

        assert split in ['train', 'val'], "split must be either 'train' or 'val'"
        self.split = split

        # data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'clip_data', 'text-to-image-2M')
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'clip_data', 'text-to-image-2M-8k') # test smaller shards
        shards = [f for f in os.listdir(data_dir) if f.startswith(f'text_to_image_{self.split}') and f.endswith('.tar')]
        shards = [os.path.join(data_dir, f) for f in shards]
        self.shards = sorted(shards)
        self.shard_count = len(self.shards)
        assert self.shard_count > 0, f"no shards found for {split} split"
        if self.shard_count == 1: 
            self.prefetch = False # don't prefetch if only one shard
        if self.verbose: 
            print(f"[dataloader-{split}.{rank}] found {self.shard_count} shards for {split} split")
            print(f"[dataloader-{split}.{rank}] using {n_proc} decoding threads")
        # set initial state
        self.reset()

    def __del__(self):
        """shutdown prefetch executor"""
        if hasattr(self, 'prefetch_executor'):
            self.prefetch_executor.shutdown(wait=False)
        
    def reset(self):
        self.next_shard(reset=True)

    def next_shard(self, reset=False):
        current_index = None
        if hasattr(self, 'shard_index'):
            current_index = self.shard_index
        new_index = 0 if reset else self.next_shard_index()
        self.shard_index = new_index
        self.current_position = self.rank * self.B
        if current_index != new_index:
            self.load_shard(prefetch=False)
        # background prefetch the next shard
        if self.prefetch:
            self.prefetch_shard()

    def next_shard_index(self):
        return (self.shard_index + 1) % self.shard_count

    def next_batch(self):
        start = self.current_position
        end = start + self.B
        images = self.images[start:end]
        labels = self.labels[start:end]
        self.step(1)
        return labels, images

    def step(self, n_step = 1):
        for i in range(n_step):
            B, world_size, rank = self.B, self.world_size, self.rank
            self.current_position += B * world_size
            # reset if next batch would be out of bounds
            next_position = self.current_position + B * world_size
            if next_position > self.shard_len:
                self.next_shard()

    def prefetch_shard(self):
        """prefetches the next shard in a background thread"""
        self.prefetch_future = self.prefetch_executor.submit(self.load_shard, prefetch=True)

    def load_shard(self, prefetch = False):
        """loads a shard, using prefetched shard if available"""
        if not prefetch and self.prefetch_future is not None:
            self.prefetch_future.result()
            tmp_images, tmp_labels = self.images, self.labels
            self.images, self.labels, self.shard_len = self.prefetch_images, self.prefetch_labels, self.prefetch_shard_len
            self.prefetch_images, self.prefetch_labels = tmp_images, tmp_labels
            self.prefetch_future = None
            if self.verbose: print(f"[dataloader-{self.split}.{self.rank}] loaded {self.shard_len} pairs from prefetched shard {self.shard_index+1}/{self.shard_count}")
        else:
            verbose = self.verbose
            shard_index = self.next_shard_index() if prefetch else self.shard_index
            shard_path = self.shards[shard_index]
            # read all files in shard into memory (raw bytes)
            with tarfile.open(shard_path, "r") as tar:
                files = sorted([m for m in tar.getmembers() if m.isfile()], key=lambda m: m.name)
                blobs = [tar.extractfile(m).read() for m in files]
            # pair image/text files (assuming matching, sequential filenames)
            blob_len = len(blobs)
            pairs = [(blobs[i], blobs[i+1]) for i in range(0, blob_len, 2)]
            shard_len = blob_len // 2
            images, labels = self.preallocate_tensors(shard_len, prefetch)
            # parallel decode images & labels (writes directly to pre-allocated tensors)
            with ThreadPoolExecutor(max_workers=self.n_proc) as ex:
                futures = [ex.submit(decode_and_write, i, pairs[i][0], pairs[i][1], images, labels) for i in range(shard_len)]
                # wait for all to complete
                for future in futures:
                    future.result()
            if prefetch:
                self.prefetch_shard_len = shard_len
                if self.verbose: print(f"[dataloader-{self.split}.{self.rank}] prefetched {self.shard_len} pairs from shard {shard_index+1}/{self.shard_count}")
            else:
                self.shard_len = shard_len
                if self.verbose: print(f"[dataloader-{self.split}.{self.rank}] loaded {self.shard_len} pairs from shard {shard_index+1}/{self.shard_count}")
    
    def preallocate_tensors(self, shard_len, prefetch):
        """uses currently allocated tensors if possible, or pre-allocates to avoid ongoing allocation overhead"""
        if not prefetch:
            if self.images is None or self.images.shape[0] != shard_len:
                self.images = torch.empty(shard_len, 3, self.img_size, self.img_size, dtype=torch.float32)
            if self.labels is None or self.labels.shape[0] != shard_len:
                self.labels = torch.empty(shard_len, self.block_size, dtype=torch.long)
            return self.images, self.labels
        else:
            if self.prefetch_images is None or self.prefetch_images.shape[0] != shard_len:
                self.prefetch_images = torch.empty(shard_len, 3, self.img_size, self.img_size, dtype=torch.float32)
            if self.prefetch_labels is None or self.prefetch_labels.shape[0] != shard_len:
                self.prefetch_labels = torch.empty(shard_len, self.block_size, dtype=torch.long)
            return self.prefetch_images, self.prefetch_labels

    def get_state(self):
        """returns state dict for checkpointing"""
        return {
            'shard_index': self.shard_index,
            'current_position': self.current_position,
        }
    
    def load_state(self, state):
        """loads state dict from checkpoint"""
        self.shard_index = state['shard_index']
        self.current_position = state['current_position']
        self.load_shard()

if __name__ == "__main__":
    import time
    from clip import TextTokenizer, TextConfig
    enc = TextTokenizer(TextConfig())
    B = 4096
    torch.manual_seed(42)
    print(f"loading & timing a preview batch with B = {B}\n-----")
    t0 = time.time()
    train_loader = DataLoader(B = B, block_size = 77, img_size = 224, rank = 0, world_size = 1, split = 'train', verbose = True)
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