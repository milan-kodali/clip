import torch
import time
import math
import argparse
import os

from clip import CLIP, TextConfig, VisionConfig
from dataloader import DataLoaderLite
from dataloader2 import DataLoader
from checkpoint_manager import save_checkpoint, update_configs_from_checkpoint, load_checkpoint_state
from logger import Logger

t0 = time.time() # tracking init time

# ------------------------------
# checkpointing args
# ------------------------------

parser = argparse.ArgumentParser(description='Train CLIP model')
parser.add_argument('--checkpoint', '-c', type=str, default=None, help='Checkpoint filename to save and load from')
args = parser.parse_args()

checkpoint_path = None
if args.checkpoint:
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'clip_data', 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)

# initialize logger
run_name = args.checkpoint if args.checkpoint else 'default'
logger = Logger(run_name=run_name)
logger.load_logs()

# ------------------------------
# set up model, training params,
# optimizer, and lr schedule 
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print(f"using device: {device}")
# Set seed across all device types
torch.manual_seed(42)
if "cuda" in device:
    torch.cuda.manual_seed_all(42)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(42)
ddp = False
master_process = True

# batch parameters
total_batch_size = 1024 # TODO: tune, probably needs to be ~1024
B = 1024 if device == "cuda" else 8 if device == "mps" else 1 
assert total_batch_size % B == 0, "total_batch_size must be divisible by B"
embed_accum_steps = total_batch_size // B

# load configs from checkpoint if available, otherwise use defaults
text_config, vision_config = TextConfig(), VisionConfig()
if checkpoint_path:
    update_configs_from_checkpoint(checkpoint_path, text_config, vision_config, device)

# data loader
train_loader = DataLoader(B=B, block_size=text_config.block_size, img_size=vision_config.img_size, rank=0, world_size=1, split = 'train', verbose = True)

# perf: use TF32 for operations on FP32s
torch.set_float32_matmul_precision('high')
# initialize model
model = CLIP(text_config, vision_config)
model = model.to(device)
# perf: compile model
if "cuda" in device:
  if master_process: print("compiling model\n-----")
  model = torch.compile(model)
  torch.cuda.synchronize() # wait for all kernels to complete
  if master_process: print("model compiled\n-----")


# set up optimizer & lr schedule with warmup & cosine decay
max_lr = 3e-4 # TODO: update
min_lr = max_lr * 0.1
img_per_epoch = 2097152
steps_per_epoch = img_per_epoch // total_batch_size # 2k steps with total_batch_size = 1024
n_epoch = 1
max_steps = steps_per_epoch * n_epoch 
optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr) # TODO: update
warmup_steps = max_steps * 0.02 # 2% warmup steps

def get_lr(step):
    # start with linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    # base case: min_lr
    return min_lr

# load checkpoint state if available
start_step = 0
if checkpoint_path:
    start_step = load_checkpoint_state(checkpoint_path, model, optimizer, train_loader, device)

# ------------------------------
# evaluate on val dataset
# ------------------------------

val_loader = DataLoader(B=B, block_size=text_config.block_size, img_size=vision_config.img_size, rank=0, world_size=1, split = 'val', verbose = True)

# tracking init time
t1 = time.time()
print(f"init time: {t1 - t0:.2f}s\n-----")

def evaluate(model, device, val_loader):
    t0 = time.time()
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        label_embs, image_embs = [], []
        for _ in range(embed_accum_steps):
            labels, images = val_loader.next_batch()
            labels, images = labels.to(device), images.to(device)
            label_emb, image_emb = model.embed(labels, images)
            label_embs.append(label_emb)
            image_embs.append(image_emb)
        label_embs = torch.cat(label_embs, dim=0)
        image_embs = torch.cat(image_embs, dim=0)
        loss = model.loss(label_embs, image_embs)
    t1 = time.time()
    dt = t1 - t0
    if master_process:
        print(f"-----\nval loss: {loss.item():.4f} | eval_time: {dt:.2f}s\n-----")
    return loss.item()

# ------------------------------
# training loop
# ------------------------------

print(f"training for {max_steps} steps")
# baseline
if start_step == 0:
    logger.log(step=0, val_loss=evaluate(model, device, val_loader))

for step in range(start_step, max_steps):
    t0 = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # label_embs, image_embs = [], []
    D = model.text_config.out_dim 
    label_embs = torch.empty((B * embed_accum_steps, D), device=device, dtype=torch.bfloat16)
    image_embs = torch.empty((B * embed_accum_steps, D), device=device, dtype=torch.bfloat16)
    for i in range(embed_accum_steps):
        labels, images = train_loader.next_batch()
        labels, images = labels.to(device), images.to(device)
        # perf: autocast to cast some ops to BF16 
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            label_emb, image_emb = model.embed(labels, images)
        label_embs[i*B:(i+1)*B] = label_emb
        image_embs[i*B:(i+1)*B] = image_emb
    # label_embs = torch.cat(label_embs, dim=0)
    # image_embs = torch.cat(image_embs, dim=0)
    # perf: autocast to cast some ops to BF16
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss = model.loss(label_embs, image_embs)
    loss.backward()
    # TODO: decide if we need gradient clipping 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    t1 = time.time()
    dt = t1 - t0
    img_per_sec = (B * embed_accum_steps) / dt
    if master_process:
        print(f"step {step} |  trn_loss: {loss.item():.4f} | norm: {norm:.4f} | lr: {lr:.2e} | dt: {dt:.2f}s | img/s: {img_per_sec:.2f}")
        if (step + 1) % 10 == 0:
            logger.log(step=step+1, train_loss=loss.item(), lr=lr, grad_norm=norm, dt=dt, imgs_per_sec=img_per_sec)
            logger.log(step=step+1, val_loss=evaluate(model, device, val_loader))
            logger.save_plot()
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, optimizer, step, train_loader, device)
                