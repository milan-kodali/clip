"""
Checkpoint utilities
"""

import os
import torch

def save_checkpoint(checkpoint_path, model, optimizer, step, train_loader, device):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'train_loader_state': train_loader.get_state(),
        'text_config': model.text_config.__dict__,
        'vision_config': model.vision_config.__dict__,
    }
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"checkpoint saved\n-----")

def update_configs_from_checkpoint(checkpoint_path, text_config, vision_config, device):
    if os.path.exists(checkpoint_path):
        print(f"loading configs from checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'text_config' in checkpoint and 'vision_config' in checkpoint:
            # update configs from saved dicts
            text_config.__dict__.update(checkpoint['text_config'])
            vision_config.__dict__.update(checkpoint['vision_config'])
            print(f"configs updated from checkpoint")

def load_checkpoint_state(checkpoint_path, model, optimizer, train_loader, device):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        train_loader.load_state(checkpoint['train_loader_state'])
        print(f"checkpoint loaded: resuming from step {step + 1}")
        return step + 1
    else:
        print(f"no checkpoint found, starting from scratch")
        return 0

def load_checkpoint_model_only(checkpoint_path, model, device):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        # Remove _orig_mod. prefix if present (from compiled models)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"checkpoint model loaded successfully")
    else:
        print(f"no checkpoint found, starting from scratch")