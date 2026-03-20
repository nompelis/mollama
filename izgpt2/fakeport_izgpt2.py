### This script creates a model file in our format by reading from the
### HuggingFace format, but it skips writing the parameter arrays. It is
### is for parsing blocks and making sure the headers are correct.

import struct
import torch
import numpy as np
#from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

# ============================================================
# CONFIG
# ============================================================

OUT_FILE = "fake.izgpt2"

MAGIC = b"IZGPT2\x00\x00"
ENDIAN_TAG = 0x4245  # 'BE'

# block types
BLK_MODEL_INFO         = 0x0001
BLK_TOKEN_EMBEDDING    = 0x0010
BLK_POSITION_EMBEDDING = 0x0011
BLK_TRANSFORMER_LAYER  = 0x0020
BLK_FINAL_LAYERNORM    = 0x0030
BLK_END                = 0x7FFF

ACT_GELU = 1

# ============================================================
# HELPERS
# ============================================================

def be_u16(x): return struct.pack(">H", x)
def be_u32(x): return struct.pack(">I", x)
def be_u64(x): return struct.pack(">Q", x)
def be_f32(x): return struct.pack(">f", x)

def write_f32_array_be(f, arr):
    arr = np.asarray(arr, dtype=np.float32)
    for v in arr.flatten():
        f.write(be_f32(float(v)))

def pad(f, n):
    f.write(b"\x00" * n)

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

# ============================================================
# FILE HEADER
# ============================================================

def write_file_header(f, block_count_placeholder=0):

    header = b""

    header += MAGIC
    header += be_u16(1)  # version major
    header += be_u16(0)  # version minor

    header += be_u16(ENDIAN_TAG)
    header += be_u16(256)

    header += be_u32(0)  # file_flags
    header += be_u64(0)  # total_file_size (fill later)
    ### Hardwire the block count for now
    block_count_placeholder= 1 + 1 + 12 + 1
    header += be_u64(block_count_placeholder)

    header += be_u64(1)  # model_flags

    header += be_u32(L)
    header += be_u32(NH)
    header += be_u32(H)
    header += be_u32(F)
    header += be_u32(CTX)
    header += be_u32(VOCAB)

    header += be_u32(0)
    header += be_u64(0)

    header += b"\x00" * 16  # checksum

    header += b"\x00" * (256 - len(header))

    f.write(header)

# ============================================================
# BLOCK HEADER
# ============================================================

def write_block_header(f, block_type, block_index, payload_size, layer_index=0xFFFFFFFF):

    header = b""

    header += be_u16(128)
    header += be_u16(block_type)
    header += be_u32(1)  # version

    header += be_u64(block_index)
    header += be_u64(payload_size)
    header += be_u64(0)  # checksum placeholder

    header += be_u32(0)  # flags
    header += be_u32(0)  # header checksum

    header += be_u32(layer_index)
    header += be_u32(0)

    header += be_u64(0)

    header += b"\x00" * 72

    assert len(header) == 128
    f.write(header)

# ============================================================
# EXPORT
# ============================================================

with open(OUT_FILE, "wb") as f:

    block_index = 0

    write_file_header(f)
    print("Wrote header")

    # --------------------------------------------------------
    # TOKEN EMBEDDING
    # --------------------------------------------------------
    Wte = sd["wte.weight"].numpy()  # (VOCAB, H)
    payload_size = Wte.size * 4

    write_block_header(f, BLK_TOKEN_EMBEDDING, block_index, payload_size)
    #write_f32_array_be(f, Wte)
    block_index += 1
    print("Wrote embeddings")

    # --------------------------------------------------------
    # POSITION EMBEDDING
    # --------------------------------------------------------
    Wpe = sd["wpe.weight"].numpy()  # (CTX, H)
    payload_size = Wpe.size * 4

    write_block_header(f, BLK_POSITION_EMBEDDING, block_index, payload_size)
    #write_f32_array_be(f, Wpe)
    block_index += 1
    print("Wrote positional encodings")

    # --------------------------------------------------------
    # TRANSFORMER LAYERS
    # --------------------------------------------------------
    for i in range(L):

        # ---- LayerNorm 1 ----
        ln1_w = sd[f"h.{i}.ln_1.weight"].numpy()
        ln1_b = sd[f"h.{i}.ln_1.bias"].numpy()
        ln1_eps = np.array([1e-5], dtype=np.float32)

        # ---- Attention ----
        W = sd[f"h.{i}.attn.c_attn.weight"].numpy()  # (H, 3H)
        b = sd[f"h.{i}.attn.c_attn.bias"].numpy()    # (3H,)

        Wq = W[:, :H].T
        Wk = W[:, H:2*H].T
        Wv = W[:, 2*H:].T

        bq = b[:H]
        bk = b[H:2*H]
        bv = b[2*H:]

        Wo = sd[f"h.{i}.attn.c_proj.weight"].numpy().T
        bo = sd[f"h.{i}.attn.c_proj.bias"].numpy()

        # ---- LayerNorm 2 ----
        ln2_w = sd[f"h.{i}.ln_2.weight"].numpy()
        ln2_b = sd[f"h.{i}.ln_2.bias"].numpy()
        ln2_eps = np.array([1e-5], dtype=np.float32)

        # ---- FFN ----
        W1 = sd[f"h.{i}.mlp.c_fc.weight"].numpy().T   # (F, H)
        b1 = sd[f"h.{i}.mlp.c_fc.bias"].numpy()

        W2 = sd[f"h.{i}.mlp.c_proj.weight"].numpy().T # (H, F)
        b2 = sd[f"h.{i}.mlp.c_proj.bias"].numpy()

        # ---- Compute payload size ----
        floats = (
            (2*H + 1) +
            4*(H*H + H) +
            (2*H + 1) +
            (F*H + F) +
            (H*F + H)
        )
        payload_size = floats * 4

        write_block_header(
            f,
            BLK_TRANSFORMER_LAYER,
            block_index,
            payload_size,
            layer_index=i
        )

        # ---- Payload ----

        # LN1
        #write_f32_array_be(f, ln1_w)
        #write_f32_array_be(f, ln1_b)
        #write_f32_array_be(f, ln1_eps)

        # Attention
        #write_f32_array_be(f, Wq)
        #write_f32_array_be(f, bq)

        #write_f32_array_be(f, Wk)
        #write_f32_array_be(f, bk)

        #write_f32_array_be(f, Wv)
        #write_f32_array_be(f, bv)

        #write_f32_array_be(f, Wo)
        #write_f32_array_be(f, bo)

        # LN2
        #write_f32_array_be(f, ln2_w)
        #write_f32_array_be(f, ln2_b)
        #write_f32_array_be(f, ln2_eps)

        # FFN (activation in header only)

        #write_f32_array_be(f, W1)
        #write_f32_array_be(f, b1)

        #write_f32_array_be(f, W2)
        #write_f32_array_be(f, b2)

        block_index += 1
        print("Wrote layer")

    # --------------------------------------------------------
    # FINAL LAYERNORM
    # --------------------------------------------------------
    ln_f_w = sd["ln_f.weight"].numpy()
    ln_f_b = sd["ln_f.bias"].numpy()
    ln_f_eps = np.array([1e-5], dtype=np.float32)

    payload_size = (2*H + 1) * 4

    write_block_header(f, BLK_FINAL_LAYERNORM, block_index, payload_size)

    #write_f32_array_be(f, ln_f_w)
    #write_f32_array_be(f, ln_f_b)
    #write_f32_array_be(f, ln_f_eps)

    block_index += 1
    print("Wrote final LayerNorm")

    # --------------------------------------------------------
    # END BLOCK
    # --------------------------------------------------------
    write_block_header(f, BLK_END, block_index, 0)

print("Export complete:", OUT_FILE)

