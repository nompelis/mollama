### Dumps biases for the Q,K,V, Wo, FFN1/2 projections and LN arrays.

from transformers import GPT2Model

import numpy as np
import os

def _write_vector(fname, vec):
    with open(fname, "w") as f:
        for v in vec:
            f.write(f"{float(v):.10e}\n")


def dump_attention_biases(sd, layer_idx, outdir="debug_attn"):

    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------------------
    # Load HuggingFace bias
    # --------------------------------------------------------
    b = sd[f"h.{layer_idx}.attn.c_attn.bias"].detach().cpu().numpy()  # (3H,)

    H = b.shape[0] // 3

    print(f"[dump] layer {layer_idx} bias: H={H}")

    # --------------------------------------------------------
    # Split Q, K, V
    # --------------------------------------------------------
    bq = b[:H]
    bk = b[H:2*H]
    bv = b[2*H:]
    bo = sd[f"h.{layer_idx}.attn.c_proj.bias"].detach().cpu().numpy()
    b1 = sd[f"h.{layer_idx}.mlp.c_fc.bias"].detach().cpu().numpy()
    b2 = sd[f"h.{layer_idx}.mlp.c_proj.bias"].detach().cpu().numpy()
    ln1_gamma = sd[f"h.{layer_idx}.ln_1.weight"].detach().cpu().numpy()
    ln1_beta  = sd[f"h.{layer_idx}.ln_1.bias"].detach().cpu().numpy()
    ln2_gamma = sd[f"h.{layer_idx}.ln_2.weight"].detach().cpu().numpy()
    ln2_beta  = sd[f"h.{layer_idx}.ln_2.bias"].detach().cpu().numpy()
    lnf_gamma = sd["ln_f.weight"].detach().cpu().numpy()
    lnf_beta  = sd["ln_f.bias"].detach().cpu().numpy()

    # --------------------------------------------------------
    # Write files
    # --------------------------------------------------------
    _write_vector(f"{outdir}/bq_L{layer_idx}.txt", bq)
    _write_vector(f"{outdir}/bk_L{layer_idx}.txt", bk)
    _write_vector(f"{outdir}/bv_L{layer_idx}.txt", bv)
    _write_vector(f"{outdir}/bo_L{layer_idx}.txt", bo)
    _write_vector(f"{outdir}/b1_L{layer_idx}.txt", b1)
    _write_vector(f"{outdir}/b2_L{layer_idx}.txt", b2)
    _write_vector(f"{outdir}/LN1g_L{layer_idx}.txt", ln1_gamma)
    _write_vector(f"{outdir}/LN1b_L{layer_idx}.txt", ln1_beta)
    _write_vector(f"{outdir}/LN2g_L{layer_idx}.txt", ln2_gamma)
    _write_vector(f"{outdir}/LN2b_L{layer_idx}.txt", ln2_beta)
    _write_vector(f"{outdir}/LNfg_L{layer_idx}.txt", lnf_gamma)
    _write_vector(f"{outdir}/LNfb_L{layer_idx}.txt", lnf_beta)

    print("[dump] wrote Q/K/V bias files")


model = GPT2Model.from_pretrained("gpt2")
sd = model.state_dict()

dump_attention_biases(sd, layer_idx=0)

