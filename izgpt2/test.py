import struct
import torch
import numpy as np
#from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


# ============================================================
# LOAD MODEL
# ============================================================

model = GPT2Model.from_pretrained("gpt2")
sd = model.state_dict()
cfg = model.config

H = cfg.n_embd
L = cfg.n_layer
NH = cfg.n_head
F = 4 * H
CTX = cfg.n_positions
VOCAB = cfg.vocab_size

print("H", H, "Layers:", L)
print("NH", NH, "F:", F)
#print("Wte shape:", Wte.shape)
#print("Layer 0 Wq shape:", Wq.shape)

