import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np

from PIL import Image
import tiktoken
from dataclasses import dataclass

# ------------------------------
# Shared Transformer Blocks
# ------------------------------

class MLP(nn.Module):
    """linear layer + non-linearity to add compute after multi-head attention"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # expand onto higher dimensional space
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # project back down to model's embedding dimensionality 

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Attention(nn.Module):
    """multiple self-attention heads in parallel"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, batched together
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.is_decoder = config.is_decoder


    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, n_embd
        # calculate query, key, value for all heads in a batch
        # C = n_head * head_size, eg n_head = 12, head_size = 64, so C = 768
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2) #(B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2) #(B, T, n_head, head_size) -> (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2) #(B, T, n_head, head_size) -> (B, n_head, T, head_size)
        
        # use flash attention instead of manually implemented attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_decoder) # (B, n_head, T, head_size)
        
        y = y.transpose(1, 2).reshape(B, T, -1) # (B, n_head, T, head_size) -> (B, T, n_head * head_size)

        y = self.c_proj(y) 
        return y

class Block(nn.Module):
    """transformer block: communication followed by computation, with resid connection (x +)"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ------------------------------
# Text Decoder (GPT-style)
# ------------------------------

@dataclass
class TextConfig:
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    out_dim: int = 512
    is_decoder: bool = True
    block_size: int = 77
    vocab_size: int = 50258 # from TextTokenizer

class TextDecoder(nn.Module):
    """GPT-style transformer decoder for text embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.proj = nn.Linear(config.n_embd, config.out_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Use pytorch default LayerNorm init

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # idx and targets are both (B, T) tensor of integers
        token_emb = self.transformer.wte(idx) # (B, T, C = n_embd)
        pos_emb = self.transformer.wpe(torch.arange(T, device=idx.device)) # (T, C = n_embd)
        x = token_emb + pos_emb # (B, T, C = n_embd)
        for block in self.transformer.h:
            x = block(x) # (B, T, C = n_embd)
        x = self.transformer.ln_f(x)
        # Find first occurrence of eot_token_id (50256)
        # TODO: don't hardcode EOT token id
        eot_positions = (idx == 50256).int().argmax(dim=-1)  # (B,)
        eot_token = x[torch.arange(x.shape[0], device=idx.device), eot_positions]  # (B, C = n_embd)
        out = self.proj(eot_token) # (B, C = out_dim)
        out = out / out.norm(dim=-1, keepdim=True) # normalize to unit length for cosine similarity

        return out

# ------------------------------
# Text Tokenizer
# ------------------------------

class TextTokenizer: 
    """ 
    tiktoken Wrapper
    Not quite CLIP tokenizer, but approximates it using GPT-2 tokenizer 
    Vocab size = GPT-2 vocab size (50257) + 1 (for new SOT token) = 50258
    """

    def __init__(self, config):
        self.enc = tiktoken.get_encoding("gpt2")
        # Special tokens
        self.eot_token = "<|endoftext|>"
        self.pad_token = self.eot_token
        self.sot_token = "<|startoftext|>"
        self.eot_token_id = 50256 # already exists in GPT-2 tokenizer
        self.pad_token_id = self.eot_token_id
        self.sot_token_id = self.eot_token_id + 1 # doesn't exist in GPT-2 tokenizer
        
        self.block_size = config.block_size

    def encode(self, text):
        tokens = [self.sot_token_id]
        text_enc = self.enc.encode(text)
        if len(text_enc) + 2 > self.block_size:
            tokens.extend(text_enc[:self.block_size - 2])
        else:
            tokens.extend(text_enc)
            if len(tokens) < self.block_size:
                tokens.extend([self.pad_token_id] * (self.block_size - 1 - len(tokens)))
        tokens.extend([self.eot_token_id])
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, ids, include_special_tokens=True):
        result = ""
        for id in ids.tolist():
            if id == self.sot_token_id:
                if include_special_tokens:
                    result += self.sot_token
            elif id == self.eot_token_id:
                if include_special_tokens:
                    result += self.eot_token
            else:
                result += self.enc.decode([id])
        return result

# ------------------------------
# Vision Encoder
# ------------------------------

@dataclass
class VisionConfig:
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 768
    out_dim: int = 512
    is_decoder: bool = False
    img_size: int = 224 # 224x224 image
    patch_size: int = 16 # 16x16 patches

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        assert self.img_size % self.patch_size == 0
        self.conv = nn.Conv2d(in_channels=3, out_channels=config.n_embd, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.conv(x) # [B, C=3, S=224, S=224] -> [B, C=768, N=14, N=14] 
        x = x.flatten(2) # [B, C=768, N=14, N=14] -> [B, C=768, N**2=T=196] 
        x = x.transpose(1,2)  #[B, C, T] -> [B, T=196, C=768] 

        return x

class VisionEncoder(nn.Module):
    """vision transformer encoder for image embeddings"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.img_size % config.patch_size == 0
        self.n_patch = (config.img_size // config.patch_size)**2 # N**2 = (224/16)**2 = 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd)) # extra learnable token
        
        self.transformer = nn.ModuleDict(dict(
            patch_emb = PatchEmbedding(config),
            pos_emb = nn.Embedding(self.n_patch + 1, config.n_embd), # +1 for cls_token
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.proj = nn.Linear(config.n_embd, config.out_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Use pytorch default LayerNorm & Conv2d init
        
    def forward(self, x):
        x = self.transformer.patch_emb(x) # [B, C=3, S=224, S=224] -> [B, T=196, C=768]
        B, T, C = x.shape

        # add cls_token to each batch
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) # [B, 197, C=768]
        # add positional embedding
        pos_emb = self.transformer.pos_emb(torch.arange(T+1, device=x.device)) # [197, C=768]
        x = x + pos_emb # [B, 197, C=768]
        for block in self.transformer.h:
            x = block(x) # [B, 197, C=768]
        x = self.transformer.ln_f(x) # [B, 197, C=768]
        cls_token = x[:, 0, :] # [B, 197, C=768] -> [B, C=768]
        out = self.proj(cls_token) # [B, C=768] -> [B, C=512]
        out = out / out.norm(dim=-1, keepdim=True) # normalize to unit length for cosine similarity

        return out

# ------------------------------
# CLIP Model
# ------------------------------

class CLIP(nn.Module):
    def __init__(self, text_config, vision_config):
        super().__init__()
        self.text_decoder = TextDecoder(text_config)
        self.vision_encoder = VisionEncoder(vision_config)
        # learnable temperature parameter (initialized to match CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def embed(self, labels, images):
        # embed labels & images
        label_emb = self.text_decoder(labels)
        image_emb = self.vision_encoder(images)
        return label_emb, image_emb
    
    def loss(self, label_emb, image_emb):
        # clamp temperature
        logit_scale = torch.clamp(self.logit_scale, max=np.log(100))
        logits = logit_scale.exp() * label_emb @ image_emb.T
        # compute contrastive loss
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2.0
        return loss

    def forward(self, labels, images):
        return self.loss(*self.embed(labels, images))


# ------------------------------
# test one forward pass
# ------------------------------

if __name__ == "__main__":

    text_config = TextConfig()
    tokenizer = TextTokenizer(text_config)
    vision_config = VisionConfig()

    # image to tensor
    itot = transforms.Compose([
    transforms.Resize((vision_config.img_size, vision_config.img_size)),
    transforms.ToTensor()
    ])
    # tensor to image
    ttoi = transforms.ToPILImage()

    # sample image loader
    def load_image(name):
        dir = "./data/sample"
        file = f"{dir}/{name}"
        return itot(Image.open(file).convert("RGB"))

    labels = torch.stack([
        tokenizer.encode("a boy and a girl"),
        tokenizer.encode("a red ball"),
        tokenizer.encode("a boy and a girl playing soccer in the park with a red ball"),
    ])

    images = torch.stack([
        load_image("1.jpg"),
        load_image("2.jpg"),
        load_image("1.jpg")
    ])

    torch.manual_seed(42)

    model = CLIP(text_config, vision_config)
    loss = model(labels, images)
    print(f"loss: {loss:.4f}")