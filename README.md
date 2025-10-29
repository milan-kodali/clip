# CLIP 

From-scratch [CLIP](https://openai.com/index/clip/)-like model for tinkering & learning 

## TODOs
1. Initial training tests & logs
2. Add evals (eg ImageNet ZeroShot)
3. Make dataloader way faster (potentially pre-save tensors)
4. Run experiments to see how much we can improve

## "Water through the Pipes" Training Runs

|          | TextDecoder | VisionEncoder |
|----------|-------------|---------------|
| n_layer  |      6      |       8       |
| n_head   |      8      |       6       |
| n_embd   |     512     |      768      |
| out_dim  |     512     |      512      |
|  block   |      77     |       -       |
| img_size |      -      |      224      |
|patch_size|      -      |       16      |

#### Run 1
*1024 pairs/batch, 0.5 epochs, 3e-4 max_lr, linear warmup for 2% of steps, then cosine decay to 10% of max_lr*


## Files
- `clip.ipynb` notebook used to sketch out & test each component 
- `clip.py` model definition 
- `prep_dataset.py` downloads [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) dataset, resizes images, and saves local shards 
- `dataloader.py` simple data loader that loads each shard tarfile into memory, and provides batches for training 
- `train.py` training script 
- `logger` simple logger to track training metrics & create graphs
- `checkpoint_manager` utility functions to save checkpoints while training and load/resume from checkpoints 
- `setup_{}.sh` scripts for env setup on  new boxes
