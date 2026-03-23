# The C loader of the GPT2 model from HuggingFace

This code uses Python and related libraries to retrieve the model weights
from HuggingFace and bring them to a file that adheres to our format. The
format is easily loadable into C/C++ programs from its binary representation
on disk.

HuggingFace's distirbution involves ```pip install``` type operations, which
tend to make things *easy* for those who want them easy at the cost of not
knowing what is going on. This code uses Python to fetch the model once, and
to store it into a binary file that can be subsequently moved around safely,
etc, while it remains readable by its accompagnying codebase.

- - -

## Model representation

The following is a representation of the GPT2 model as it is percieved when
processing tokens in practice (this is the essential order). Multiple layers
of the **transformer block** exist (e.g. 12 for GPT2 "0.1B parameters" model),
with **residual connections** around them. This is an **encoder only**
architecture.

According to the **machine learing literature**, tokens are represented as
**rows** (that is "_row vectors_"), which is the opposite convention from
the linear algebra literature. This convention is closer to what memory looks
like in **row-major** operations in a programming language like **C**, and how
we store matrices as blocks of memory. The index that unrolls fastest is
the _second index_, such that a *matvec* operation like the following makes
sense:
```
for (int i = 0; i < ROWS; ++i) {
    real_t r=0.0;
    for (int j = 0; j < COLS; ++j)
        r += mat[i * COLS + j] * vec[j]
    out[i] = r;
}
```
The benefits to such a layout are mostly optimization related.

### Token embeddings

Embeddings are vectors of **768 32-bit floats** for GPT2, and there are 50257
vectors for the embeddings; this size is stored in the variable ```vocab_size```
in our format. The number 768 is the **hidden dimension**, is denoted ny
**H**, and it is stored in the variable ```n_embd``` in our data structures.

When loaded in memory (in row-major format), this block of memory is:
````
------------------------------------------------------------
TOKEN EMBEDDING PAYLOAD
------------------------------------------------------------
  Wte[vocab_size * H]
````

### Positional encodings / position embeddings

These embeddings have the same dimension as the token embeddings (H), but
they are sized by the **context length**, which is 1024 for this model.
The context length is denoted by **C**; this is stored in the variable
```n_ctx``` in our data structures.

When loaded in memory (in row-major format), the block of memory is:
```
------------------------------------------------------------
POSITION EMBEDDING PAYLOAD
------------------------------------------------------------
  Wpe[n_ctx * H]
```

### Transformer blocks

Each transformer block processes tokens through several layers:
```
LayerNorm --> Attention --> Projection --> LayerNorm --> FeedForward
```
For a **single head** attention block, the layout of the parameters is
the same as for a **multi-head** attention block. The difference is that
the Q,K,V row-vectors that are produced are partitioned in the H dimension
into equal parts across heads. (This is such that each head can perform
scaled dot-product attention scoring based on fewer features.) The only
implication in terms of memory is that there are **num_head** C x C
blocks of 32-bit floats that are used during inference when multi-head
atteion is threaded in parallel, but the parameter count is the same.

The **feed-forward** layer "_expands_" the hidden dimension from H to
**F = 4 x H** in the GPT2 model. The expanded vector passes element-wise through
the activation function, and then is recompressed via another affine operation
down to H hidden dimensions.

The layout of a single transformer block looks like this:
```
------------------------------------------------------------
TRANSFORMER LAYER PAYLOAD ORDER
------------------------------------------------------------
LN1:
  gamma[H]
  beta[H]
  epsilon[1]

ATTENTION:
  Wq[H*H]
  bq[H]

  Wk[H*H]
  bk[H]

  Wv[H*H]
  bv[H]

  Wo[H*H]
  bo[H]

LN2:
  gamma[H]
  beta[H]
  epsilon[1]

FFN:
  (activation scalar params may be needed)

  W1[F*H]
  b1[F]

  W2[H*F]
  b2[H]
```

### Final LayerNorm

A final "LayerNorm" operation is performed to each token after it has gone
through the stack of transformer blocks. It is exactly like the LayerNorm
operations inside the transformer block, with identical parameter counts.
It looks like this:
```
------------------------------------------------------------
FINAL LAYERNORM PAYLOAD
------------------------------------------------------------
  gamma[H]
  beta[H]
  epsilon[1]
```
Like all LayerNorm blocks, it uses an "_epsilon_" (32-bit float representing
a small value) to avoid numerical issues.

### Logits

Once the **final token** in the context has been processed through the
transformer layers, a single embedding/token vector is produced. This vector
is mapped through an expansion to the vocabulary size to produce **logits**.
In the GPT2 model, the same vector as the _embeddings_ is used to transform
back "_from token to vocabulary_" by producing logits; this is because the
embeddings are intuitively related to the inverse transformation. The matrix
used to produce the logits is the transpose of the matrix of embedding
vectors ```Wte```.

- - -

## The C memory structures for the model

### Direct (naive) model loading

The model can be loaded into memory block-by-block. That is, instead of
allocating a large memory block and assigning pointers to certain spots
in it, we allocate each block (array) individually. The data structures
to form the model are like this:
```
// ============================================================
//  MODEL
// ============================================================

typedef struct {
    float epsilon;
    float *gamma;   // H
    float *beta;    // H
} ao_ln_t;

typedef struct {
    float *Wq;      // H*H
    float *bq;      // H
    float *Wk;      // H*H
    float *bk;      // H
    float *Wv;      // H*H
    float *bv;      // H
    float *Wo;      // H*H
    float *bo;      // H
} ao_attn_t;

typedef struct {
    uint32_t act_kind;
    uint32_t act_param_count;
    float   *act_params;

    float *W1;      // F*H
    float *b1;      // F
    float *W2;      // H*F
    float *b2;      // H
} ao_ffn_t;

typedef struct {
    ao_ln_t   ln1;
    ao_attn_t attn;
    ao_ln_t   ln2;
    ao_ffn_t  ffn;
} ao_layer_t;

typedef struct {
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_embd;
    uint32_t n_ctx;
    uint32_t vocab_size;
    uint32_t ffn_dim;

    float *wte;     // vocab_size * H
    float *wpe;     // n_ctx * H

    ao_layer_t *layers;

    ao_ln_t ln_f;

    float *lm_head; // optional vocab_size * H
    float *lm_bias; // optional vocab_size
} ao_gpt2_t;

```

### Flattened model loading

The model can be loaded into a **POSIX shared memory segment** in a flattened
array, with individual blocks are serialized in the same order that adhere
to ***non-packed C structs***. That is, the structs and the array of layer
structs have the layout that the host prefers for computation. However, at
the top of the segment the (packed) struct that is the header of the file
from which the model was loaded is found. The leyout in memory looks like
this:
```
// ============================================================
//  MODEL LAYOUT FOR A SHARED MEMORY UNIFIED SEGMENT
// ============================================================
// Pointers inside structs are relative from the top of the segment
//
//    izgpt2_file_header_t;
//    ao_gpt2_t;
//    ao_layer_t [L];
//    float [V * H];             // token embedings
//    float [V * H];             // position embedings
//    float [L * layer_size];    // transformer blocks
//    float [1 + 2*H];           // final LayerNorm arrays
//
```

A convenient struct is used to handle operations via the shared memory
segment, especially when the process tht did not create it attaches:
```
typedef struct {
    int id;               // OS level POSIX shared memory segment ID
    struct shmid_ds ds;   // data from last operation on segment
    void* base;           // base pointer in local memory space
    size_t off[8];        // offsets for different structs
    ao_gpt2_t *model;     // Once loaded, pointers to structs are local.
                          // Non-owner processes need to recreate payload
                          // pointers from the offsets.
} sm_gpt2_t;
```
The owner process of the shared memory segment interprets pointers as local
references. Any attaching process _must_ call the function ```build_shmem_model()```
to localize the model in its memory space; this happens automatically when
the ```sm_gpt2_t *smattach_model(int id)``` function is called.

The following constants show the order of the offsets and access the
individual offset slots in the ```off[]``` array.
```
enum {
       GPT2_OFFSET_HEADER,
       GPT2_OFFSET_MODEL,
       GPT2_OFFSET_LAYER_ARRAY,
       GPT2_OFFSET_TOKEMB,
       GPT2_OFFSET_POSEMB,
       GPT2_OFFSET_TRANS_BLOCKS,
       GPT2_OFFSET_LNF,
       GPT2_OFFSET_TOTAL
};
```


- - -
IN 2026/03/23
