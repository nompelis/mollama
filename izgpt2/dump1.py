### Dumps the 1st, 2nd, and last vectors of token embeddings and pos. encodings

from transformers import GPT2Model

import numpy as np
import os

def _write_vector(fname, vec):
    with open(fname, "w") as f:
        for v in vec:
            f.write(f"{float(v):.10e}\n")


# ============================================================
# TOKEN EMBEDDINGS (Wte)
# ============================================================

def dump_token_embeddings(sd, outdir="debug_wte"):

    os.makedirs(outdir, exist_ok=True)

    Wte = sd["wte.weight"].detach().cpu().numpy()  # (vocab, H)

    vocab_size, H = Wte.shape

    print(f"[dump] token embeddings: vocab={vocab_size}, H={H}")

    # indices
    idx0 = 0
    idx1 = 1
    idxN = vocab_size - 1

    _write_vector(f"{outdir}/wte_0.txt", Wte[idx0])
    _write_vector(f"{outdir}/wte_1.txt", Wte[idx1])
    _write_vector(f"{outdir}/wte_last.txt", Wte[idxN])

    print("[dump] wrote token embedding vectors")


# ============================================================
# POSITION EMBEDDINGS (Wpe)
# ============================================================

def dump_pos_embeddings(sd, outdir="debug_wpe"):

    os.makedirs(outdir, exist_ok=True)

    Wpe = sd["wpe.weight"].detach().cpu().numpy()  # (ctx, H)

    n_ctx, H = Wpe.shape

    print(f"[dump] position embeddings: ctx={n_ctx}, H={H}")

    idx0 = 0
    idx1 = 1
    idxN = n_ctx - 1

    _write_vector(f"{outdir}/wpe_0.txt", Wpe[idx0])
    _write_vector(f"{outdir}/wpe_1.txt", Wpe[idx1])
    _write_vector(f"{outdir}/wpe_last.txt", Wpe[idxN])

    print("[dump] wrote position embedding vectors")


model = GPT2Model.from_pretrained("gpt2")
sd = model.state_dict()

dump_token_embeddings(sd)
dump_pos_embeddings(sd)
