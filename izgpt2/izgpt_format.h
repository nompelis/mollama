#ifndef IZGPT2_FORMAT_H
#define IZGPT2_FORMAT_H

#include <stdint.h>
#include <sys/shm.h>

// ============================================================
//  GLOBAL CONSTANTS
// ============================================================

#define IZGPT2_FILE_HEADER_SIZE   256
#define IZGPT2_BLOCK_HEADER_SIZE  128

#define IZGPT2_MAGIC "IZGPT2"

// ============================================================
//  ENUMS
// ============================================================

// Block types (strict file order, no skipping)
typedef enum {
    IZGPT2_BLK_MODEL_INFO         = 0x0001,
    IZGPT2_BLK_TOKEN_EMBEDDING    = 0x0010,
    IZGPT2_BLK_POSITION_EMBEDDING = 0x0011,
    IZGPT2_BLK_TRANSFORMER_LAYER  = 0x0020,
    IZGPT2_BLK_FINAL_LAYERNORM    = 0x0030,
    IZGPT2_BLK_END                = 0x7FFF
} izgpt2_block_type_t;


// Activation types (fixed, no custom for now)
typedef enum {
    IZGPT2_ACT_NONE       = 0,
    IZGPT2_ACT_GELU       = 1,
    IZGPT2_ACT_RELU       = 2,
    IZGPT2_ACT_SILU       = 3,
    IZGPT2_ACT_TANH       = 4,
    IZGPT2_ACT_LEAKY_RELU = 5,
    IZGPT2_ACT_ELU        = 6
} izgpt2_activation_t;

// ============================================================
//  FILE HEADER (256 bytes)
// ============================================================

typedef struct __attribute__((packed)) {

    // 0
    char     magic[8];           // "IZGPT2\0\0"

    // 8
    uint16_t version_major;
    uint16_t version_minor;

    // 12
    uint16_t endian_tag;        // 0x4245 = 'BE'
    uint16_t header_size;       // 256

    // 16
    uint32_t file_flags;

    // 20
    uint64_t total_file_size;

    // 28
    uint64_t block_count;

    // 36
    uint64_t model_flags;

    // 44
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_embd;            // H
    uint32_t ffn_dim;           // F
    uint32_t n_ctx;
    uint32_t vocab_size;

    // 68
    uint32_t reserved0;

    // 72
    uint64_t reserved1;

    // 80
    uint8_t  file_checksum[16]; // placeholder

    // 96
    uint8_t  padding[160];      // pad to 256

} izgpt2_file_header_t;

// ============================================================
//  GENERIC BLOCK HEADER (128 bytes)
// ============================================================

typedef struct __attribute__((packed)) {

    // 0
    uint16_t header_size;       // 128
    uint16_t block_type;

    // 4
    uint32_t block_version;

    // 8
    uint64_t block_index;

    // 16
    uint64_t payload_size;

    // 24
    uint64_t payload_checksum;  // placeholder

    // 32
    uint32_t block_flags;

    // 36
    uint32_t header_checksum;   // placeholder

    // 40
    uint32_t layer_index;       // or 0xFFFFFFFF
    uint32_t reserved0;

    // 48
    uint64_t reserved1;

    // 56
    uint8_t  type_specific[72]; // fills to 128

} izgpt2_block_header_t;


// ============================================================
//  TYPE-SPECIFIC BLOCK HEADERS
// ============================================================

// -------------------------------
// MODEL INFO BLOCK
// (mostly redundant but explicit)
// -------------------------------
typedef struct __attribute__((packed)) {

    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_embd;
    uint32_t n_ctx;
    uint32_t vocab_size;
    uint32_t ffn_dim;

    uint32_t reserved[10];

} izgpt2_model_info_block_t;

// -------------------------------
// TOKEN EMBEDDING BLOCK
// -------------------------------
typedef struct __attribute__((packed)) {

    uint32_t vocab_size;
    uint32_t n_embd;

    uint32_t reserved[14];

} izgpt2_token_embedding_block_t;


// -------------------------------
// POSITION EMBEDDING BLOCK
// -------------------------------
typedef struct __attribute__((packed)) {

    uint32_t n_ctx;
    uint32_t n_embd;

    uint32_t reserved[14];

} izgpt2_position_embedding_block_t;

// -------------------------------
// TRANSFORMER LAYER BLOCK
// -------------------------------
typedef struct __attribute__((packed)) {

    uint32_t layer_index;

    uint32_t n_embd;        // H
    uint32_t n_head;
    uint32_t head_dim;      // = H / n_head
    uint32_t ffn_dim;       // F

    // LN1
    uint32_t ln1_type;      // always LayerNorm
    uint32_t ln1_param_count; // always 1

    // Attention
    uint32_t attn_type;     // causal self-attn
    uint32_t attn_flags;

    // LN2
    uint32_t ln2_type;
    uint32_t ln2_param_count; // always 1

    // FFN
    uint32_t ffn_type;

    // Activation
    uint32_t activation_type;
    float    activation_param1;
    float    activation_param2;

    uint32_t reserved[6];

} izgpt2_transformer_layer_block_t;


// -------------------------------
// FINAL LAYERNORM BLOCK
// -------------------------------
typedef struct __attribute__((packed)) {

    uint32_t n_embd;
    uint32_t norm_type;
    uint32_t param_count;   // always 1

    uint32_t reserved[13];

} izgpt2_final_ln_block_t;

// ============================================================
//  PAYLOAD LAYOUT DEFINITIONS
// ============================================================

/*
All matrices are stored row-major:
  for (row)
    for (col)
      emit(M[row][col])

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
  (activation params already in header)

  W1[F*H]
  b1[F]

  W2[H*F]
  b2[H]

------------------------------------------------------------
TOKEN EMBEDDING PAYLOAD
------------------------------------------------------------
  Wte[vocab_size * H]

------------------------------------------------------------
POSITION EMBEDDING PAYLOAD
------------------------------------------------------------
  Wpe[n_ctx * H]

------------------------------------------------------------
FINAL LAYERNORM PAYLOAD
------------------------------------------------------------
  gamma[H]
  beta[H]
  epsilon[1]

*/

// ============================================================
//  HELPER MACROS (OPTIONAL)
// ============================================================

#define IZGPT2_LN_PARAM_COUNT 1

// Compute expected floats in a transformer layer
static inline uint64_t
izgpt2_layer_float_count(uint32_t H, uint32_t F)
{
    return
        (2*H + 1) +        // LN1 (gamma, beta arrays & epsilon scalar)
        4*(H*H + H) +      // attention (attn matrix & softmax temp vec
        (2*H + 1) +        // LN2
        (F*H + F) +        // W1 + b1
        (H*F + H);         // W2 + b2
}


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



// --- API ---

//  Structures for storing the model in a shared memory segment

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

typedef struct {
    int id;
    struct shmid_ds ds;
    void* base;
    size_t off[8];
    ao_gpt2_t *model;
} sm_gpt2_t;


// The API call to load the model in memory

ao_gpt2_t *load_model(const char *path);

// The API call to release all memory associated with the model

void destroy_model(ao_gpt2_t *m);


// The API call to load the model in a new shared memory segment

sm_gpt2_t *smload_model(const char *path);


#endif

