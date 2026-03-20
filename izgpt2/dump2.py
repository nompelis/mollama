### Dumps Q,K,V "vectors of the matrix" (vector that dots with embed)
### for 1st, 2nd, last elements of output vectors

from transformers import GPT2Model

import numpy as np
import os

def _write_vector(fname, vec):
    with open(fname, "w") as f:
        for v in vec:
            f.write(f"{float(v):.10e}\n")


def dump_attention_columns(sd, layer_idx, outdir="debug_attn"):

    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------------------
    # Load HuggingFace tensors
    # --------------------------------------------------------
    W = sd[f"h.{layer_idx}.attn.c_attn.weight"].detach().cpu().numpy()  # (H, 3H)
    b = sd[f"h.{layer_idx}.attn.c_attn.bias"].detach().cpu().numpy()    # (3H,)

    H = W.shape[0]

    print(f"[dump] layer {layer_idx} attention: H={H}")

    # --------------------------------------------------------
    # Split Q, K, V & Wo (HuggingFace layout)
    # --------------------------------------------------------
    Wq_hf = W[:, :H]
    Wk_hf = W[:, H:2*H]
    Wv_hf = W[:, 2*H:]
    Wo_hf = sd[f"h.{layer_idx}.attn.c_proj.weight"].detach().cpu().numpy()


    # --------------------------------------------------------
    # Transpose to YOUR layout (H, H) of (F, H) etc
    # --------------------------------------------------------
    Wq = Wq_hf.T
    Wk = Wk_hf.T
    Wv = Wv_hf.T
    Wo = Wo_hf.T
    W1 = sd[f"h.{layer_idx}.mlp.c_fc.weight"].detach().cpu().numpy().T
    W2 = sd[f"h.{layer_idx}.mlp.c_proj.weight"].detach().cpu().numpy().T

    # --------------------------------------------------------
    # Columns to extract
    # --------------------------------------------------------
    cols = [0, 1, H - 1]
    cols_W1 = [0, 1, W1.shape[1] - 1]  # columns in H dimension
    cols_W2 = [0, 1, W2.shape[1] - 1]  # columns in F dimension

    # --------------------------------------------------------
    # Dump helpers
    # --------------------------------------------------------
    def dump_matrix_columns(M, name):
        for c in cols:
            vec = M[:, c]   # column c
            fname = f"{outdir}/{name}_col{c}.txt"
            _write_vector(fname, vec)

    def dump_matrix_columns2(M, name):
        for c in cols_W1:
            vec = M[:, c]   # column c
            fname = f"{outdir}/{name}_col{c}.txt"
            _write_vector(fname, vec)

    def dump_matrix_columns3(M, name):
        for c in cols_W2:
            vec = M[:, c]   # column c
            fname = f"{outdir}/{name}_col{c}.txt"
            _write_vector(fname, vec)

    # --------------------------------------------------------
    # Write files
    # --------------------------------------------------------
    dump_matrix_columns(Wq, f"Wq_L{layer_idx}")
    dump_matrix_columns(Wk, f"Wk_L{layer_idx}")
    dump_matrix_columns(Wv, f"Wv_L{layer_idx}")
    dump_matrix_columns(Wo, f"Wo_L{layer_idx}")
    dump_matrix_columns2(W1, f"W1_L{layer_idx}")
    dump_matrix_columns3(W2, f"W2_L{layer_idx}")

    print("[dump] wrote Wq/Wk/Wv/Wo,FFw1/FFw2 column files")


model = GPT2Model.from_pretrained("gpt2")
sd = model.state_dict()

#dump_attention_columns(sd, layer_idx=0)
dump_attention_columns(sd, layer_idx=11)

