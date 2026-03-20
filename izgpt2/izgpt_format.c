#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "utils.h"
#include "izgpt_format.h"


// Functions to do endian-ness conversions

static inline uint16_t bswap16(uint16_t x)
{
    uint16_t y;
    izgpt_util_ConvertEndian( &y, &x, 2 );
    return y;
}

static inline uint32_t bswap32(uint32_t x)
{
    uint32_t y;
    izgpt_util_ConvertEndian( &y, &x, 4 );
    return y;
}

static inline uint64_t bswap64(uint64_t x)
{
    uint64_t y;
    izgpt_util_ConvertEndian( &y, &x, 8 );
    return y;
}

static inline float bswapf(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    v.u = bswap32(v.u);
    return v.f;
}

void swap_f32_array(float *data, size_t n)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Doing endian-swap for %ld 32-bit floats\n", n );
#endif
    for (size_t i = 0; i < n; i++) data[i] = bswapf(data[i]);
}


// Function to read binary data from a _binary_ stream

int read_exact(FILE *f, void *ptr, size_t n)
{
    size_t iret = fread(ptr, 1, n, f);
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Read %ld bytes \n", iret );
    perror("read_exact");
    if( iret != n ) {
        fprintf( stdout, " [Error]  Read %ld instead of %ld bytes\n", iret, n );
    }
#endif
    return iret == n;
}


// Function to read the header of the IZGPT2 format

int load_file_header(FILE *f, izgpt2_file_header_t *h)
{
    if( !read_exact(f, h, sizeof(*h)) ) {
        fprintf( stdout, " [Error]  Could not read header from stream\n" );
        return -1;
    }

    // endian swap from BE on disk to local endian-ness
    h->version_major = bswap16(h->version_major);
    h->version_minor = bswap16(h->version_minor);

    h->endian_tag    = bswap16(h->endian_tag);
    h->header_size   = bswap16(h->header_size);

    h->file_flags    = bswap32(h->file_flags);
    h->total_file_size = bswap64(h->total_file_size);
    h->block_count     = bswap64(h->block_count);
    h->model_flags     = bswap64(h->model_flags);

    h->n_layer     = bswap32(h->n_layer);
    h->n_head      = bswap32(h->n_head);
    h->n_embd      = bswap32(h->n_embd);
    h->ffn_dim     = bswap32(h->ffn_dim);
    h->n_ctx       = bswap32(h->n_ctx);
    h->vocab_size  = bswap32(h->vocab_size);
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loaded header (v %d.%d)\n",
             h->version_major, h->version_minor );
    fprintf( stdout, " Endian tag: 0x%.4x (16-bit), size: %d\n",
             h->endian_tag, h->header_size );
    fprintf( stdout, " File flags: %.8x \n", h->file_flags );
    fprintf( stdout, " Total file size: %ld \n", h->total_file_size );
    fprintf( stdout, " Block count: %ld \n", h->block_count );
    fprintf( stdout, " Model flags: %.16lx \n", h->model_flags );
    fprintf( stdout, " Transformer layers: %d \n", h->n_layer );
    fprintf( stdout, " Transformer heads: %d \n", h->n_head );
    fprintf( stdout, " Model size (embed. dim.): %d \n", h->n_embd );
    fprintf( stdout, " Feed-forward size: %d \n", h->ffn_dim );
    fprintf( stdout, " Context length (tokens): %d \n", h->n_ctx );
    fprintf( stdout, " Vocabulary size: %d \n", h->vocab_size );
#endif

    // validate
    if (memcmp(h->magic, "IZGPT2", 6) != 0) {
        fprintf( stdout, " [Error]  Not a IZGPT2 formatted file\n" );
        return -1;
    }
    if (h->header_size != IZGPT2_FILE_HEADER_SIZE) {
        fprintf( stdout, " [Error]  Header has the wrong size \n" );
        return -1;
    }
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Model file's header is valid\n" );
#endif

    return 0;
}


// Function to do memory allocation for the top-level structs of the model

ao_gpt2_t *alloc_model(const izgpt2_file_header_t *h)
{
    ao_gpt2_t *m = calloc(1, sizeof(*m));
    if(!m) {
        fprintf( stdout, " [Error]  Could not create toplevel struct\n" );
        return NULL;
    }

    m->n_layer = h->n_layer;
    m->n_head  = h->n_head;
    m->n_embd  = h->n_embd;
    m->ffn_dim = h->ffn_dim;
    m->n_ctx   = h->n_ctx;
    m->vocab_size = h->vocab_size;

    m->wte = NULL;
    m->wpe = NULL;

    m->layers = calloc(h->n_layer, sizeof(ao_layer_t));
    if(!m->layers) {
        fprintf( stdout, " [Error]  Could not create layers struct array\n" );
        free(m);
        return NULL;
    }
    for (int l=0; l < m->n_layer; l++) {
        m->layers[l].ln1.epsilon = -99.9e99;
        m->layers[l].ln1.gamma = NULL;
        m->layers[l].ln1.beta = NULL;

        m->layers[l].attn.Wq = NULL;
        m->layers[l].attn.bq = NULL;
        m->layers[l].attn.Wk = NULL;
        m->layers[l].attn.bk = NULL;
        m->layers[l].attn.Wv = NULL;
        m->layers[l].attn.bv = NULL;
        m->layers[l].attn.Wo = NULL;
        m->layers[l].attn.bo = NULL;

        m->layers[l].ln2.epsilon = -99.9e99;
        m->layers[l].ln2.gamma = NULL;
        m->layers[l].ln2.beta = NULL;

        m->layers[l].ffn.act_kind = IZGPT2_ACT_NONE;
        m->layers[l].ffn.act_param_count = 0;
        m->layers[l].ffn.act_params = NULL;
        m->layers[l].ffn.W1 = NULL;
        m->layers[l].ffn.b1 = NULL;
        m->layers[l].ffn.W2 = NULL;
        m->layers[l].ffn.b2 = NULL;
    }

    m->ln_f.epsilon = -99.9e99;
    m->ln_f.gamma = NULL;
    m->ln_f.beta = NULL;

    m->lm_head = NULL;
    m->lm_bias = NULL;

    return m;
}


// Function to be a header-block loader

int load_block_header(FILE *f, izgpt2_block_header_t *b)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading block header\n" );
#endif
//printf("Sizeof: %ld\n", sizeof(*b));//HACK
    if (!read_exact(f, b, sizeof(*b))) {
       fprintf( stdout, " [Error]  Could not read 128-byte block\n" );
       return -1;
    }

    b->header_size = bswap16(b->header_size);
    b->block_type  = bswap16(b->block_type);
#ifdef _DEBUG_
    switch(b->block_type) {
    case(IZGPT2_BLK_MODEL_INFO):
        fprintf( stdout, " [DEBUG]  Block is \"MODEL_INFO\"\n");
        break;
    case(IZGPT2_BLK_TOKEN_EMBEDDING):
        fprintf( stdout, " [DEBUG]  Block is \"TOKEN EMBEDDINGS\"\n");
        break;
    case(IZGPT2_BLK_POSITION_EMBEDDING):
        fprintf( stdout, " [DEBUG]  Block is \"POSITION ENCODINGS\"\n");
        break;
    case(IZGPT2_BLK_TRANSFORMER_LAYER):
        fprintf( stdout, " [DEBUG]  Block is \"TRANSFORMER\"\n");
        break;
    case(IZGPT2_BLK_FINAL_LAYERNORM):
        fprintf( stdout, " [DEBUG]  Block is \"LAYERNORM\"\n");
        break;
    default:
        fprintf( stdout, " [Error]  Unknown block type: %d\n", b->block_type );
    }
#endif

    b->block_version = bswap32(b->block_version);
    b->block_index   = bswap64(b->block_index);
    b->payload_size  = bswap64(b->payload_size);

    b->block_flags   = bswap32(b->block_flags);
    b->layer_index   = bswap32(b->layer_index);

    return 0;
}


// Function to be the token embeddings loader

int load_token_embedding(FILE *f, ao_gpt2_t *m, uint64_t payload_size)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading token embeddings\n" );
#endif
    size_t count = m->vocab_size * m->n_embd;

    m->wte = malloc(count * sizeof(float));
    if( !m->wte ) {
        fprintf( stdout, " [Error]  Could not allocate %ld bytes\n", count );
        return -1;
    }

    if (!read_exact(f, m->wte, payload_size)) {
        fprintf( stdout, " [Error]  Could not read parameters\n" );
        return 1;
    }

    swap_f32_array(m->wte, count);

    return 0;
}


// Function to be the positional encodings loader

int load_pos_embedding(FILE *f, ao_gpt2_t *m, uint64_t payload_size)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading positional encodings\n" );
#endif
    size_t count = m->n_ctx * m->n_embd;

    m->wpe = malloc(count * sizeof(float));

    if (!read_exact(f, m->wpe, payload_size)) return -1;

    swap_f32_array(m->wpe, count);

    return 0;
}


// Function to be a layer's parameter loader

int load_layer(FILE *f, ao_gpt2_t *m, int i, uint64_t payload_size)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading layer: %d/%d\n", i, m->n_layer );
#endif
    ao_layer_t *L = &m->layers[i];

    int H = m->n_embd;
    int F = m->ffn_dim;

    /* LN1 */
    L->ln1.gamma = malloc(H*sizeof(float));
    L->ln1.beta  = malloc(H*sizeof(float));

    read_exact(f, L->ln1.gamma, H*sizeof(float));
    read_exact(f, L->ln1.beta,  H*sizeof(float));
    read_exact(f, &L->ln1.epsilon, sizeof(float));

    /* Attention */
    size_t HH = H*H;

    #define LOAD_VEC(ptr, n) do { \
        ptr = malloc((n)*sizeof(float)); \
        read_exact(f, ptr, (n)*sizeof(float)); \
    } while(0)

    LOAD_VEC(L->attn.Wq, HH);
    LOAD_VEC(L->attn.bq, H);

    LOAD_VEC(L->attn.Wk, HH);
    LOAD_VEC(L->attn.bk, H);

    LOAD_VEC(L->attn.Wv, HH);
    LOAD_VEC(L->attn.bv, H);

    LOAD_VEC(L->attn.Wo, HH);
    LOAD_VEC(L->attn.bo, H);

    /* LN2 */
    L->ln2.gamma = malloc(H*sizeof(float));
    L->ln2.beta  = malloc(H*sizeof(float));

    read_exact(f, L->ln2.gamma, H*sizeof(float));
    read_exact(f, L->ln2.beta,  H*sizeof(float));
    read_exact(f, &L->ln2.epsilon, sizeof(float));

    /* FFN */
    LOAD_VEC(L->ffn.W1, F*H);
    LOAD_VEC(L->ffn.b1, F);

    LOAD_VEC(L->ffn.W2, H*F);
    LOAD_VEC(L->ffn.b2, H);

    #undef LOAD_VEC

    /* endian swap everything */
    swap_f32_array(L->ln1.gamma, H);
    swap_f32_array(L->ln1.beta,  H);
    L->ln1.epsilon = bswapf(L->ln1.epsilon);

    swap_f32_array(L->attn.Wq, HH);
    swap_f32_array(L->attn.bq, H);

    swap_f32_array(L->attn.Wk, HH);
    swap_f32_array(L->attn.bk, H);

    swap_f32_array(L->attn.Wv, HH);
    swap_f32_array(L->attn.bv, H);

    swap_f32_array(L->attn.Wo, HH);
    swap_f32_array(L->attn.bo, H);

    swap_f32_array(L->ln2.gamma, H);
    swap_f32_array(L->ln2.beta,  H);
    L->ln2.epsilon = bswapf(L->ln2.epsilon);

    swap_f32_array(L->ffn.W1, F*H);
    swap_f32_array(L->ffn.b1, F);

    swap_f32_array(L->ffn.W2, H*F);
    swap_f32_array(L->ffn.b2, H);

    return 0;
}


// Function to be the final LayerNorm parameters loaded

int load_final_ln(FILE *f, ao_gpt2_t *m, uint64_t payload_size)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading final LayerNorm\n" );
#endif
    int H = m->n_embd;

    m->ln_f.gamma = malloc(H*sizeof(float));
    m->ln_f.beta  = malloc(H*sizeof(float));

    read_exact(f, m->ln_f.gamma, H*sizeof(float));
    read_exact(f, m->ln_f.beta,  H*sizeof(float));
    read_exact(f, &m->ln_f.epsilon, sizeof(float));

    swap_f32_array(m->ln_f.gamma, H);
    swap_f32_array(m->ln_f.beta,  H);
    m->ln_f.epsilon = bswapf(m->ln_f.epsilon);

    return 0;
}


// Function to be the main model loader

ao_gpt2_t *load_model(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf( stdout, " [Error]  Could not open file: \"%s\"\n", path );
        return NULL;
    }

    izgpt2_file_header_t fh;
    if (load_file_header(f, &fh) != 0) {
        fclose(f);
        return NULL;
    }

    ao_gpt2_t *m = alloc_model(&fh);
    if (!m) {
       fclose(f);
       return NULL;
    }

    for (uint64_t i = 0; i < fh.block_count; i++) {
#ifdef _DEBUG_
        fprintf( stdout, " [DEBUG]  About to read block: %ld \n", i );
#endif

        izgpt2_block_header_t bh;
        if (load_block_header(f, &bh) != 0) return NULL;

        switch (bh.block_type) {

        case IZGPT2_BLK_TOKEN_EMBEDDING:
            load_token_embedding(f, m, bh.payload_size);
        break;

        case IZGPT2_BLK_POSITION_EMBEDDING:
            load_pos_embedding(f, m, bh.payload_size);
        break;

        case IZGPT2_BLK_TRANSFORMER_LAYER:
            load_layer(f, m, bh.layer_index, bh.payload_size);
        break;

        case IZGPT2_BLK_FINAL_LAYERNORM:
            load_final_ln(f, m, bh.payload_size);
        break;

        case IZGPT2_BLK_END:
        break;

        default:
            fprintf( stdout, " [Error]  Unknown block type: %d\n",
                     bh.block_type);
            return NULL;
        }
    }

    fclose(f);
    return m;
}

