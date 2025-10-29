# CLIP 

CLIP model for tinkering & learning 

## TODOs
1. 

## Tests & results 



## Files
- `clip.ipynb` notebook used to sketch out & test each component 
- `clip.py` model definition 
- `prep_dataset.py` downloads [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) dataset, resizes images, and saves local shards 
- `dataloader.py` simple data loader that loads each shard tarfile into memory, and provides batches for training 
- `train.py` training script 
- `logger` simple logger to track training metrics & create graphs
- `checkpoint_manager` utility functions to save checkpoints while training and load/resume from checkpoints 
- `setup_{}.sh` scripts for env setup on  new boxes
