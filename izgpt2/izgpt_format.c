#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/shm.h>
#include <stddef.h>
#include <stdalign.h>
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
    if( iret != n ) {
        perror("read_exact");
        fprintf( stdout, " [Error]  Read %ld instead of %ld bytes\n", iret, n );
    }
#endif
    return iret == n;
}


// Function to read the header of the IZGPT2 format

int load_file_header(FILE *f, izgpt2_file_header_t *h)
{
    if (!read_exact(f, h, sizeof(*h))) {
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

int load_token_embedding(FILE *f, ao_gpt2_t *m, uint64_t payload_size, int iop)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading token embeddings\n" );
#endif
    size_t count = m->vocab_size * m->n_embd;

    if (!iop) {
        m->wte = malloc(count * sizeof(float));
        if (!m->wte) {
            fprintf( stdout, " [Error]  Could not allocate Wte \n" );
            return -1;
        }
    }

    if (!read_exact(f, m->wte, payload_size)) {
        fprintf( stdout, " [Error]  Could not read Wte\n" );
        return 1;
    }

    swap_f32_array(m->wte, count);

    return 0;
}


// Function to be the positional encodings loader

int load_pos_embedding(FILE *f, ao_gpt2_t *m, uint64_t payload_size, int iop)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading positional encodings\n" );
#endif
    size_t count = m->n_ctx * m->n_embd;

    if (iop==0) {
        m->wpe = malloc(count * sizeof(float));
        if (!m->wpe) {
            fprintf( stdout, " [Error]  Could not allocate Wpe \n" );
            return -1;
        }
    }

    if (!read_exact(f, m->wpe, payload_size)) {
        fprintf( stdout, " [Error]  Could not read Wpe\n" );
        return 1;
    }

    swap_f32_array(m->wpe, count);

    return 0;
}


// Function to be a layer's parameter loader

int load_layer(FILE *f, ao_gpt2_t *m, int i, uint64_t payload_size, int iop)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading layer: %d/%d\n", i, m->n_layer );
#endif
    ao_layer_t *L = &m->layers[i];

    int H = m->n_embd;
    int F = m->ffn_dim;

    // LN1
    if (iop==0) {
        L->ln1.gamma = malloc(H*sizeof(float));
        L->ln1.beta  = malloc(H*sizeof(float));
        if (!L->ln1.gamma || !L->ln1.beta) {
            fprintf( stdout, " [Error]  Allocation of LN_1 failed\n" );
            return -1;
        }
    }

    if (!read_exact(f, L->ln1.gamma, H*sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN1 gamma\n" );
        return 1;
    }
    if (!read_exact(f, L->ln1.beta,  H*sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN1 beta\n" );
        return 1;
    }
    if (!read_exact(f, &L->ln1.epsilon, sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN1 epsilon \n" );
        return 1;
    }

    // Attention
    size_t HH = H*H;

    #define LOAD_VEC(ptr, n) do { \
        if (iop==0 ) { \
            ptr = malloc((n)*sizeof(float)); \
            if (!ptr) { \
                fprintf( stdout, " [Error]  Allocation in atten. failed\n" ); \
                return -1; \
            } \
        } \
        if (!read_exact(f, ptr, (n)*sizeof(float))) { \
            fprintf( stdout, " [Error]  Could not read data in atten.\n" ); \
            return 1; \
        } \
    } while(0)

    LOAD_VEC(L->attn.Wq, HH);
    LOAD_VEC(L->attn.bq, H);

    LOAD_VEC(L->attn.Wk, HH);
    LOAD_VEC(L->attn.bk, H);

    LOAD_VEC(L->attn.Wv, HH);
    LOAD_VEC(L->attn.bv, H);

    LOAD_VEC(L->attn.Wo, HH);
    LOAD_VEC(L->attn.bo, H);

    // LN2
    if (iop==0) {
        L->ln2.gamma = malloc(H*sizeof(float));
        L->ln2.beta  = malloc(H*sizeof(float));
        if (!L->ln2.gamma || !L->ln2.beta) {
            fprintf( stdout, " [Error]  Allocation of LN_2 failed\n" );
            return -1;
        }
    }

    if (!read_exact(f, L->ln2.gamma, H*sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN2 gamma\n" );
        return 1;
    }
    if (!read_exact(f, L->ln2.beta,  H*sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN2 beta\n" );
        return 1;
    }
    if (!read_exact(f, &L->ln2.epsilon, sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN2 epsilon\n" );
        return 1;
    }

    // FFN
    LOAD_VEC(L->ffn.W1, F*H);
    LOAD_VEC(L->ffn.b1, F);

    LOAD_VEC(L->ffn.W2, H*F);
    LOAD_VEC(L->ffn.b2, H);

    #undef LOAD_VEC

    // endian swap everything
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

int load_final_ln(FILE *f, ao_gpt2_t *m, uint64_t payload_size, int iop)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Loading final LayerNorm\n" );
#endif
    int H = m->n_embd;

    if (iop==0) {
        m->ln_f.gamma = malloc(H*sizeof(float));
        m->ln_f.beta  = malloc(H*sizeof(float));
        if (!m->ln_f.gamma || !m->ln_f.beta) {
            fprintf( stdout, " [Error]  Allocation of LN_f failed\n" );
            return -1;
        }
    }

    if (!read_exact(f, m->ln_f.gamma, H*sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN_f gamma\n" );
        return 1;
    }
    if (!read_exact(f, m->ln_f.beta,  H*sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN_f beta\n" );
        return 1;
    }
    if (!read_exact(f, &m->ln_f.epsilon, sizeof(float))) {
        fprintf( stdout, " [Error]  Could not read LN_f epsilon\n" );
        return 1;
    }

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

    int ierr=0;
    for (uint64_t i = 0; i < fh.block_count && ierr==0; i++) {
#ifdef _DEBUG_
        fprintf( stdout, " [DEBUG]  About to read block: %ld \n", i );
#endif

        izgpt2_block_header_t bh;
        if (load_block_header(f, &bh) != 0) {
            fclose(f);
            destroy_model(m);
            return NULL;
        }

        switch (bh.block_type) {

        case IZGPT2_BLK_TOKEN_EMBEDDING:
            ierr = load_token_embedding(f, m, bh.payload_size, 0);
        break;

        case IZGPT2_BLK_POSITION_EMBEDDING:
            ierr = load_pos_embedding(f, m, bh.payload_size, 0);
        break;

        case IZGPT2_BLK_TRANSFORMER_LAYER:
            ierr = load_layer(f, m, bh.layer_index, bh.payload_size, 0);
        break;

        case IZGPT2_BLK_FINAL_LAYERNORM:
            ierr = load_final_ln(f, m, bh.payload_size, 0);
        break;

        case IZGPT2_BLK_END:
        break;

        default:
            fprintf( stdout, " [Error]  Unknown block type: %d\n",
                     bh.block_type);
            ierr=1;
        }
    }

    fclose(f);
    if (ierr) {
        destroy_model(m);
        return NULL;
    }
    return m;
}


// Function to release all memory associated with the model

void destroy_model(ao_gpt2_t *m)
{
    if (m->wte) free(m->wte);
    if (m->wpe) free(m->wpe);

    if (m->layers) {
        for (int l=0; l < m->n_layer; l++) {
            ao_layer_t* lp = &( m->layers[l] );

            if (lp->ln1.gamma) free(lp->ln1.gamma);
            if (lp->ln1.beta) free(lp->ln1.beta);

            if (lp->attn.Wq) free(lp->attn.Wq);
            if (lp->attn.bq) free(lp->attn.bq);
            if (lp->attn.Wk) free(lp->attn.Wk);
            if (lp->attn.bk) free(lp->attn.bk);
            if (lp->attn.Wv) free(lp->attn.Wv);
            if (lp->attn.bv) free(lp->attn.bv);
            if (lp->attn.Wo) free(lp->attn.Wo);
            if (lp->attn.bo) free(lp->attn.bo);

            if (lp->ln2.gamma) free(lp->ln2.gamma);
            if (lp->ln2.beta) free(lp->ln2.beta);

            if (lp->ffn.W1) free(lp->ffn.W1);
            if (lp->ffn.b1) free(lp->ffn.b1);
            if (lp->ffn.W2) free(lp->ffn.W2);
            if (lp->ffn.b2) free(lp->ffn.b2);
        }
    }

    if (m->layers) free(m->layers);

    if (m->ln_f.gamma) free(m->ln_f.gamma);
    if (m->ln_f.beta) free(m->ln_f.beta);

    if (m->lm_head) free(m->lm_head);
    if (m->lm_bias) free(m->lm_bias);

    free(m);
}


// --------------- loading to shared memory segment ----------------

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


// Helper function to do alignments
static inline size_t align_up(size_t x, size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}


// Function to do memory allocation for the model in a shared memory segment
// (The object that this function returns stores relative offsets instead of
// pointers; pointer stores an offset as a positive integer.)

sm_gpt2_t *smalloc_model(const izgpt2_file_header_t *h)
{
    if(!h) {
        fprintf( stdout, " [Error]  Pointer to header is null\n");
        return NULL;
    }

    // object to potentially return...
    sm_gpt2_t *smgpt = calloc(1, sizeof(*smgpt));
    if(!smgpt) {
        fprintf( stdout, " [Error]  Could not create toplevel struct\n" );
        return NULL;
    }

    ao_gpt2_t *m = calloc(1, sizeof(*m));
    if(!m) {
        fprintf( stdout, " [Error]  Could not create model struct\n" );
        free(smgpt);
        return NULL;
    }
    smgpt->model = m;

    uint32_t L          = m->n_layer = h->n_layer;
    uint32_t NH         = m->n_head  = h->n_head;
    uint32_t H          = m->n_embd  = h->n_embd;
    uint32_t F          = m->ffn_dim = h->ffn_dim;
    uint32_t n_ctx      = m->n_ctx   = h->n_ctx;
    uint32_t vocab_size = m->vocab_size = h->vocab_size;

    m->wte = NULL;
    m->wpe = NULL;

    m->layers = NULL;

    m->ln_f.epsilon = -99.9e99;
    m->ln_f.gamma = NULL;
    m->ln_f.beta = NULL;

    m->lm_head = NULL;
    m->lm_bias = NULL;


    // count size incrementally
    size_t cursor=0;
    size_t* off = smgpt->off;

    // add the two main structs (header and model toplevel)
    cursor = align_up( cursor, alignof(izgpt2_file_header_t) );
    off[GPT2_OFFSET_HEADER] = cursor;
    cursor += sizeof(izgpt2_file_header_t);

    cursor = align_up( cursor, alignof(ao_gpt2_t) );
    off[GPT2_OFFSET_MODEL] = cursor;
    cursor += sizeof(ao_gpt2_t);

    // add the array of layers
    cursor = align_up( cursor, alignof(ao_layer_t) );
    off[GPT2_OFFSET_LAYER_ARRAY] = cursor;
    cursor += sizeof(ao_layer_t) * L;

    // from this point on payloads are blocks of floats, so allign once
    cursor = align_up( cursor, alignof(float) );

    // add token and position embeddings
    off[GPT2_OFFSET_TOKEMB] = cursor;
    cursor += sizeof(float) * vocab_size * H;

    off[GPT2_OFFSET_POSEMB] = cursor;
    cursor += sizeof(float) * n_ctx * H;

    // per-layer 32-bit payloads cummulative size
    size_t layer_size = izgpt2_layer_float_count(H, F);

    // add size for transformer blocks
    off[GPT2_OFFSET_TRANS_BLOCKS] = cursor;
    cursor += sizeof(float) * layer_size * L;

    // add size for final LayerNorm block
    off[GPT2_OFFSET_LNF] = cursor;
    cursor += sizeof(float) * (1 + 2*H);

    // end of segment
    off[GPT2_OFFSET_TOTAL] = cursor;

#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Shared memory pointer layout\n" );
    fprintf( stdout, "  Header: @ %ld  s %ld \n",
             off[GPT2_OFFSET_HEADER], sizeof(izgpt2_file_header_t) );
    fprintf( stdout, "  Model toplevel: @ %ld  s %ld \n",
             off[GPT2_OFFSET_MODEL], sizeof(ao_gpt2_t) );
    fprintf( stdout, "  Layers: @ %ld  s %ld \n",
             off[GPT2_OFFSET_LAYER_ARRAY], sizeof(ao_layer_t) * L );
    fprintf( stdout, "  Token embeddings: @ %ld  s %ld \n",
             off[GPT2_OFFSET_TOKEMB], sizeof(float) * vocab_size * H );
    fprintf( stdout, "  Position embeddings: @ %ld  s %ld \n",
             off[GPT2_OFFSET_POSEMB], sizeof(float) * n_ctx * H );
    fprintf( stdout, "  Transformer blocks: @ %ld  s %ld \n",
             off[GPT2_OFFSET_TRANS_BLOCKS], sizeof(float) * layer_size * L );
    fprintf( stdout, "  Final LayerNorm: @ %ld  s %ld \n",
             off[GPT2_OFFSET_LNF], sizeof(float) * (1 + 2*H) );
    fprintf( stdout, "  Total: @ %ld \n", off[GPT2_OFFSET_TOTAL] );
#endif

    // create the shared memory segment
    struct shmid_ds ds;
    int id = shmget(IPC_PRIVATE, cursor, IPC_CREAT | 0600);
    if (id < 0) {
        fprintf( stdout, " [Error]  Could not create shared memory segment\n" );
        free(m);
        free(smgpt);
        return NULL;
#ifdef _DEBUG_
    } else {
        fprintf( stdout, " [DEBUG]  Created mem segment with ID: %d\n",id);
#endif
    }
    smgpt->id = id;

    // attach to the memory segment
    void *base = shmat(id, NULL, 0);
    if (base == (void*) -1) {
        // this is unusual... but attempt to mark segment to be deleted
        shmctl( id, IPC_RMID, &ds );
        fprintf( stdout, " [Error]  Could not attach to segment: %d\n", id );
        fprintf( stdout, "   Manually remove with: 'ipcrm shm %d'\n", id );
        free(m);
        free(smgpt);
        return NULL;
    } else {
        fprintf( stdout, " [Info]  Attached to mem segment with ID: %d\n",id);
    }
    smgpt->base = base;

    // flag segment as "ready to clean" (is gone when process dettaches/exists)
    if (shmctl( id, IPC_RMID, &ds ) == -1) {
        fprintf( stdout, " [Error]  Could flag segment for deletion\n" );
        fprintf( stdout, "   Manually remove with: 'ipcrm shm %d'\n", id );
        // this is too unusual; we do not have to exit...
        // ...but cleaning needs to be done manually!
    }
    memcpy( &smgpt->ds, &ds, sizeof(ds) );

    if (shmctl( id, IPC_STAT, &ds ) == -1) {
        fprintf( stdout, " [Error]  Could retrieve segment's status\n" );
        fprintf( stdout, "   Manually remove with: 'ipcrm shm %d'\n", id );
        // this is too unusual; we do not have to exit...
#ifdef _DEBUG_
    } else {
         printf("   struct ipc_perm shm_perm; (NOT SHOWN) \n");
         printf("   size_t   shm_segsz; %ld\n", (long) ds.shm_segsz );
         printf("   time_t   shm_atime; %ld\n", (long) ds.shm_atime );
         printf("   time_t   shm_dtime; %ld\n", (long) ds.shm_dtime );
         printf("   time_t   shm_ctime; %ld\n", (long) ds.shm_ctime );
         printf("   pid_t    shm_cpid;  %ld (PID of creator)\n", (long) ds.shm_cpid );
         printf("   pid_t    shm_lpid;  %ld (PID of last shmat)\n", (long) ds.shm_lpid );
         printf("   shmatt_t shm_nattch; %d\n", (int) ds.shm_nattch );
         fprintf( stdout, " Manually remove with: 'ipcrm shm %d'\n", id );
#endif
    }


    // copy header to memory
    void* ptr = (void*) SHM_PTR( base, off[GPT2_OFFSET_HEADER] );
    memcpy( ptr, h, sizeof(*h) );

    // make memory attachments
    m->layers = (ao_layer_t*) off[GPT2_OFFSET_LAYER_ARRAY];    // relative
    m->layers = (ao_layer_t*) SHM_PTR( base, m->layers );      // local

    float* f_sm = (float*) SHM_PTR( base, off[GPT2_OFFSET_TOKEMB] );
    m->wte = f_sm;

    f_sm = (float*) SHM_PTR( base, off[GPT2_OFFSET_POSEMB] );
    m->wpe = f_sm;

    f_sm = (float*) SHM_PTR( base, off[GPT2_OFFSET_TRANS_BLOCKS] );
    for (int l=0; l < m->n_layer; l++) {
        ao_layer_t* lp = &( m->layers[l] );

        lp->ln1.epsilon = -99.9e99;
        lp->ln1.gamma = f_sm + 1;
        lp->ln1.beta  = f_sm + 1 + H;
        f_sm += 1 + 2*H;

        lp->attn.Wq = f_sm;
        lp->attn.bq = f_sm +   H*H;
        lp->attn.Wk = f_sm +   H*H +   H;
        lp->attn.bk = f_sm + 2*H*H +   H;
        lp->attn.Wv = f_sm + 2*H*H + 2*H;
        lp->attn.bv = f_sm + 3*H*H + 2*H;
        lp->attn.Wo = f_sm + 3*H*H + 3*H;
        lp->attn.bo = f_sm + 4*H*H + 3*H;
        f_sm += 4*(H*H + H);

        lp->ln2.epsilon = -99.9e99;
        lp->ln2.gamma = f_sm + 1;
        lp->ln2.beta  = f_sm + 1 + H;
        f_sm += 1 + 2*H;

        lp->ffn.act_kind = IZGPT2_ACT_NONE;
        lp->ffn.act_param_count = 0;
        lp->ffn.act_params = NULL;
        lp->ffn.W1 = f_sm;
        lp->ffn.b1 = f_sm + F*H;
        lp->ffn.W2 = f_sm + F*H + F;
        lp->ffn.b2 = f_sm + F*H + F + H*F;
        f_sm += F*H + F + H*F + H;
    }

    f_sm = (float*) SHM_PTR( base, off[GPT2_OFFSET_LNF] );
    m->ln_f.epsilon = -99.9e99;
    *f_sm = m->ln_f.epsilon;      // real copy
    m->ln_f.gamma = f_sm + 1;
    m->ln_f.beta  = f_sm + 1 + H;
    f_sm += 1 + 2*H;

#ifdef _DEBUG_
{   // Set every element of the arrays to something specific to the array
    for (int n=0; n < vocab_size*H; ++n) m->wte[n] = 1.1;
    for (int n=0; n < n_ctx*H; ++n) m->wpe[n] = 1.2;

    for (int l=0; l < m->n_layer; l++) {
        ao_layer_t* lp = &( m->layers[l] );

        lp->ln1.epsilon = 1111.111;
        for (int n=0; n < H; ++n) lp->ln1.gamma[n] = 100 + l + 0.1;
        for (int n=0; n < H; ++n) lp->ln1.beta[n] = 100 + l + 0.2;

        for (int n=0; n < H*H; ++n) lp->attn.Wq[n] = 200 + l + 0.1;
        for (int n=0; n < H  ; ++n) lp->attn.bq[n] = 200 + l + 0.2;
        for (int n=0; n < H*H; ++n) lp->attn.Wk[n] = 200 + l + 0.3;
        for (int n=0; n < H  ; ++n) lp->attn.bk[n] = 200 + l + 0.4;
        for (int n=0; n < H*H; ++n) lp->attn.Wv[n] = 200 + l + 0.5;
        for (int n=0; n < H  ; ++n) lp->attn.bv[n] = 200 + l + 0.6;
        for (int n=0; n < H*H; ++n) lp->attn.Wo[n] = 200 + l + 0.7;
        for (int n=0; n < H  ; ++n) lp->attn.bo[n] = 200 + l + 0.8;

        lp->ln2.epsilon = 3333.333;
        for (int n=0; n < H; ++n) lp->ln2.gamma[n] = 300 + l + 0.1;
        for (int n=0; n < H; ++n) lp->ln2.beta[n] = 300 + l + 0.2;

        for (int n=0; n < F*H; ++n) lp->ffn.W1[n] = 400 + l + 0.1;
        for (int n=0; n < F  ; ++n) lp->ffn.b1[n] = 400 + l + 0.2;
        for (int n=0; n < H*F; ++n) lp->ffn.W2[n] = 400 + l + 0.3;
        for (int n=0; n < H  ; ++n) lp->ffn.b2[n] = 400 + l + 0.4;
    }

    m->ln_f.epsilon = 5555.555;
    float* tmp = (float*) SHM_PTR( base, off[GPT2_OFFSET_LNF] );
    *tmp = m->ln_f.epsilon;      // real copy
    for (int n=0; n < H; ++n) m->ln_f.gamma[n] = 5.1;;
    for (int n=0; n < H; ++n) m->ln_f.beta[n] = 5.2;

    // Print first and last element of every array (looking for overlaps)
    print_diagnostics(m);
}
#endif

    // write the initial model toplevel struct to the memory segment
    // (after any model loading this needs to remain current)
    ptr = (void*) SHM_PTR( base, off[GPT2_OFFSET_MODEL] );
    memcpy( ptr, m, sizeof(*m) );

    return smgpt;
}


// Function to be the main model loader when using a shared memory segment

sm_gpt2_t *smload_model(const char *path)
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

    sm_gpt2_t *sm = smalloc_model(&fh);
//printf("PREMATURE RETURN (to not load adata)\n"); return sm;//HACK
    if (!sm) {
       fclose(f);
       return NULL;
    }

    ao_gpt2_t *m = sm->model;

    int ierr=0;
    for (uint64_t i = 0; i < fh.block_count && ierr==0; i++) {
#ifdef _DEBUG_
        fprintf( stdout, " [DEBUG]  About to read block: %ld \n", i );
#endif

        izgpt2_block_header_t bh;
        if (load_block_header(f, &bh) != 0) {
            fclose(f);
            free(sm->model);
            free(sm);
            return NULL;
        }

        switch (bh.block_type) {

        case IZGPT2_BLK_TOKEN_EMBEDDING:
            ierr = load_token_embedding(f, m, bh.payload_size, 1);
        break;

        case IZGPT2_BLK_POSITION_EMBEDDING:
            ierr = load_pos_embedding(f, m, bh.payload_size, 1);
        break;

        case IZGPT2_BLK_TRANSFORMER_LAYER:
            ierr = load_layer(f, m, bh.layer_index, bh.payload_size, 1);
#ifdef _DEBUG_
            if (ierr==0)
            fprintf( stdout, " [DEBUG]  Epsilon for LN1/2: %g, %g \n",
                     m->layers[bh.layer_index].ln1.epsilon,
                     m->layers[bh.layer_index].ln2.epsilon );
#endif
        break;

        case IZGPT2_BLK_FINAL_LAYERNORM:
            ierr = load_final_ln(f, m, bh.payload_size, 1);
#ifdef _DEBUG_
            if (ierr==0)
            fprintf( stdout, " [DEBUG]  Epsilon for LN_f: %g \n",
                     m->ln_f.epsilon );
#endif
            if (ierr==0) {
                float* tmp = (float*)
                             SHM_PTR( sm->base, sm->off[GPT2_OFFSET_LNF] );
                *tmp = m->ln_f.epsilon;      // real copy
            }
        break;

        case IZGPT2_BLK_END:
        break;

        default:
            fprintf( stdout, " [Error]  Unknown block type: %d\n",
                     bh.block_type);
            ierr=1;
        }
    }

    fclose(f);
    if (ierr) {
        free(sm->model);
        free(sm);
        return NULL;
    }

    // on success, update the model toplevel struct to the memory segment
    memcpy( (void*) SHM_PTR( sm->base, sm->off[GPT2_OFFSET_MODEL] ),
             sm->model, sizeof(*m) );

    return sm;
}


// Function to build the pointers of a shared memory block that stores a model
// The base pointer is what the pointers to data will be built with respoect to.
// Optionally provide the offsets array if you need it filled, or send null.
//
// When the "0x01" bit in "flags" is set, "m->layers" will point to a newly
// allocated array. When the "0x02" bit is set, the epsilons in the present
// struct will be coied from their locations in the memory segments; this
// implies that the memory segment is attached and the read will not fail;
// this operation only makes sense if "0x01" is also set, otherwise it errors.
// (This is useful when the calling process is meant to attach to a
// memory segment owned by a different process, and consequently the layers
// array will have invalid references.

int build_shmem_model(ao_gpt2_t *m, size_t* off, const void* base,
                      uint8_t flags)
{
    if (!m) {
        fprintf( stdout, " [Error]  Model pointer is null\n" );
        return 1;
    }

    if ((flags & 0x02) && !(flags & 0x01)) {
        fprintf( stdout, " [Error]  Bit 2 canot be set when bit 1 is unset!\n");
        return 2;
    }

    uint32_t L          = m->n_layer;
    uint32_t NH         = m->n_head;
    uint32_t H          = m->n_embd;
    uint32_t F          = m->ffn_dim;
    uint32_t n_ctx      = m->n_ctx;
    uint32_t vocab_size = m->vocab_size;
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Building Shared memory pointers\n" );
    fprintf( stdout, " Transformer layers: %d \n", L);
    fprintf( stdout, " Transformer heads: %d \n", NH);
    fprintf( stdout, " Model size (embed. dim.): %d \n", H);
    fprintf( stdout, " Feed-forward size: %d \n", F);
    fprintf( stdout, " Context length (tokens): %d \n", n_ctx);
    fprintf( stdout, " Vocabulary size: %d \n", vocab_size);
#endif

    // count size incrementally
    size_t cursor=0;
    size_t off_[8];
    if (!off) off = off_;     // user internal array when one is not provided

    // add the two main structs (header and model toplevel)
    cursor = align_up( cursor, alignof(izgpt2_file_header_t) );
    off[GPT2_OFFSET_HEADER] = cursor;
    cursor += sizeof(izgpt2_file_header_t);

    cursor = align_up( cursor, alignof(ao_gpt2_t) );
    off[GPT2_OFFSET_MODEL] = cursor;
    cursor += sizeof(ao_gpt2_t);

    // add the array of layers
    cursor = align_up( cursor, alignof(ao_layer_t) );
    off[GPT2_OFFSET_LAYER_ARRAY] = cursor;
    cursor += sizeof(ao_layer_t) * L;

    // from this point on payloads are blocks of floats, so allign once
    cursor = align_up( cursor, alignof(float) );

    // add token and position embeddings
    off[GPT2_OFFSET_TOKEMB] = cursor;
    cursor += sizeof(float) * vocab_size * H;

    off[GPT2_OFFSET_POSEMB] = cursor;
    cursor += sizeof(float) * n_ctx * H;

    // per-layer 32-bit payloads cummulative size
    size_t layer_size = izgpt2_layer_float_count(H, F);

    // add size for transformer blocks
    off[GPT2_OFFSET_TRANS_BLOCKS] = cursor;
    cursor += sizeof(float) * layer_size * L;

    // add size for final LayerNorm block
    off[GPT2_OFFSET_LNF] = cursor;
    cursor += sizeof(float) * (1 + 2*H);

    // end of segment
    off[GPT2_OFFSET_TOTAL] = cursor;
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Shared memory pointer layout\n" );
    fprintf( stdout, "  Header: @ %ld  s %ld \n",
             off[GPT2_OFFSET_HEADER], sizeof(izgpt2_file_header_t) );
    fprintf( stdout, "  Model toplevel: @ %ld  s %ld \n",
             off[GPT2_OFFSET_MODEL], sizeof(ao_gpt2_t) );
    fprintf( stdout, "  Layers: @ %ld  s %ld \n",
             off[GPT2_OFFSET_LAYER_ARRAY], sizeof(ao_layer_t) * L );
    fprintf( stdout, "  Token embeddings: @ %ld  s %ld \n",
             off[GPT2_OFFSET_TOKEMB], sizeof(float) * vocab_size * H );
    fprintf( stdout, "  Position embeddings: @ %ld  s %ld \n",
             off[GPT2_OFFSET_POSEMB], sizeof(float) * n_ctx * H );
    fprintf( stdout, "  Transformer blocks: @ %ld  s %ld \n",
             off[GPT2_OFFSET_TRANS_BLOCKS], sizeof(float) * layer_size * L );
    fprintf( stdout, "  Final LayerNorm: @ %ld  s %ld \n",
             off[GPT2_OFFSET_LNF], sizeof(float) * (1 + 2*H) );
    fprintf( stdout, "  Total: @ %ld \n", off[GPT2_OFFSET_TOTAL] );
#endif


    // make memory attachments
    m->layers = (ao_layer_t*) off[GPT2_OFFSET_LAYER_ARRAY];    // relative
    m->layers = (ao_layer_t*) SHM_PTR( base, m->layers );      // local
    ao_layer_t* sm_layers = m->layers;    // keep reference for copy
    if (flags & 0x01) {
        m->layers = (ao_layer_t*) malloc(sizeof(*(m->layers)) * L);
        if (!m->layers) {
            fprintf( stdout, " [Error]  Could not allocate layers array\n" );
            return -1;
        }
    }

    float* f_sm = (float*) SHM_PTR( base, off[GPT2_OFFSET_TOKEMB] );
    m->wte = f_sm;

    f_sm = (float*) SHM_PTR( base, off[GPT2_OFFSET_POSEMB] );
    m->wpe = f_sm;

    f_sm = (float*) SHM_PTR( base, off[GPT2_OFFSET_TRANS_BLOCKS] );
    for (int l=0; l < m->n_layer; l++) {
        ao_layer_t* lp = &( m->layers[l] );

        if (flags & 0x02) {
            lp->ln1.epsilon = sm_layers[l].ln1.epsilon;
        }
        lp->ln1.gamma = f_sm + 1;
        lp->ln1.beta  = f_sm + 1 + H;
        f_sm += 1 + 2*H;

        lp->attn.Wq = f_sm;
        lp->attn.bq = f_sm +   H*H;
        lp->attn.Wk = f_sm +   H*H +   H;
        lp->attn.bk = f_sm + 2*H*H +   H;
        lp->attn.Wv = f_sm + 2*H*H + 2*H;
        lp->attn.bv = f_sm + 3*H*H + 2*H;
        lp->attn.Wo = f_sm + 3*H*H + 3*H;
        lp->attn.bo = f_sm + 4*H*H + 3*H;
        f_sm += 4*(H*H + H);

        if (flags & 0x02) {
            lp->ln2.epsilon = sm_layers[l].ln2.epsilon;
        }
        lp->ln2.gamma = f_sm + 1;
        lp->ln2.beta  = f_sm + 1 + H;
        f_sm += 1 + 2*H;

        lp->ffn.act_kind = IZGPT2_ACT_NONE;
        lp->ffn.act_param_count = 0;
        lp->ffn.act_params = NULL;
        lp->ffn.W1 = f_sm;
        lp->ffn.b1 = f_sm + F*H;
        lp->ffn.W2 = f_sm + F*H + F;
        lp->ffn.b2 = f_sm + F*H + F + H*F;
        f_sm += F*H + F + H*F + H;
    }

    f_sm = (float*) SHM_PTR( base, off[GPT2_OFFSET_LNF] );
    if (flags & 0x02) {
        m->ln_f.epsilon = *f_sm;
    }
    m->ln_f.gamma = f_sm + 1;
    m->ln_f.beta  = f_sm + 1 + H;
    f_sm += 1 + 2*H;

    return 0;
}


// Function to attach to a shared memory segment

sm_gpt2_t *smattach_model(int id)
{
    sm_gpt2_t *sm = malloc(sizeof(*sm));
    ao_gpt2_t *m = malloc(sizeof(*m));
    if (!sm || !m) {
        if (sm) free(sm);
        if (m) free(m);
        return NULL;
    }
    size_t* off = sm->off;
    sm->model = m;

    sm->id = id;
    sm->base = shmat(id, NULL, 0);
    if (sm->base == (void*) -1) {
        fprintf( stdout, " Error]  Could not attach to shmem with ID: %d\n",id);
        free(sm);
        free(m);
        return NULL;
    }

    if (shmctl( id, IPC_STAT, &sm->ds ) == -1) {
        fprintf( stdout, " [Error]  Could retrieve segment's status. Fatal.\n");
        free(sm);
        free(m);
        return NULL;
    } else {
#ifdef _DEBUG_
         printf("   struct ipc_perm shm_perm; (NOT SHOWN) \n");
         printf("   size_t   shm_segsz; %ld\n", (long) sm->ds.shm_segsz );
         printf("   time_t   shm_atime; %ld\n", (long) sm->ds.shm_atime );
         printf("   time_t   shm_dtime; %ld\n", (long) sm->ds.shm_dtime );
         printf("   time_t   shm_ctime; %ld\n", (long) sm->ds.shm_ctime );
         printf("   pid_t    shm_cpid;  %ld (PID of creator)\n", (long) sm->ds.shm_cpid );
         printf("   pid_t    shm_lpid;  %ld (PID of last shmat)\n", (long) sm->ds.shm_lpid );
         printf("   shmatt_t shm_nattch; %d\n", (int) sm->ds.shm_nattch );
#endif
    }

    // attach to header struct
    izgpt2_file_header_t* h = (izgpt2_file_header_t*) sm->base;

    // validate
    if (memcmp(h->magic, "IZGPT2", 6) != 0) {
        fprintf( stdout, " [Error]  Not a IZGPT2 formatted file\n" );
        shmdt(sm->base);
        free(sm);
        free(m);
        return NULL;
#ifdef _DEBUG_
    } else {
       fprintf( stdout, " [DEBUG]  Magic (hdr) \"%.8s\"\n", (char*) sm->base );
#endif
    }

    // build pointers
    m->n_layer = h->n_layer;
    m->n_head  = h->n_head;
    m->n_embd  = h->n_embd;
    m->ffn_dim = h->ffn_dim;
    m->n_ctx   = h->n_ctx;
    m->vocab_size = h->vocab_size;
    build_shmem_model(sm->model, sm->off, sm->base, 0x01 | 0x02);
#ifdef _DEBUG_
    // printing offsets
    fprintf( stdout, " [DEBUG]  Offsets:\n" );
    fprintf( stdout, "   Header off: %ld \n", off[GPT2_OFFSET_HEADER] );
    fprintf( stdout, "   Model off: %ld \n", off[GPT2_OFFSET_MODEL] );
    fprintf( stdout, "   Blocks off: %ld \n", off[GPT2_OFFSET_TRANS_BLOCKS] );
    print_diagnostics(m);
#endif

#ifdef _DEBUG_
    // lift the model struct from the shared memory segment
    m = (ao_gpt2_t*) SHM_PTR( sm->base, off[GPT2_OFFSET_MODEL] );
    fprintf( stdout, " [DEBUG]  Lifted model struct contents \n" );
    fprintf( stdout, " Transformer layers: %d \n", m->n_layer);
    fprintf( stdout, " Transformer heads: %d \n", m->n_head);
    fprintf( stdout, " Model size (embed. dim.): %d \n", m->n_embd);
    fprintf( stdout, " Feed-forward size: %d \n", m->ffn_dim);
    fprintf( stdout, " Context length (tokens): %d \n", m->n_ctx);
    fprintf( stdout, " Vocabulary size: %d \n", m->vocab_size);
    fprintf( stdout, " [DEBUG]  Showing \"epsilon\" of final LayerNorm: %g\n",
             m->ln_f.epsilon );

    // lift 1st layer's epsilon directly...
    ao_layer_t* lp =
          (ao_layer_t*) SHM_PTR( sm->base, off[GPT2_OFFSET_TRANS_BLOCKS] );
    fprintf( stdout, " [DEBUG]  Lifted LN1.epsilon: %g \n", lp->ln1.epsilon );
#endif

    return sm;
}


// Function to provide diagnostics by printing 1st and last elements

void print_diagnostics(ao_gpt2_t *m)
{
    uint32_t L          = m->n_layer;
  //uint32_t NH         = m->n_head;
    uint32_t H          = m->n_embd;
    uint32_t F          = m->ffn_dim;
    uint32_t n_ctx      = m->n_ctx;
    uint32_t vocab_size = m->vocab_size;

    fprintf( stdout, " [DEBUG]  Printing first/last elements of arrays\n" );
    fprintf( stdout, " Embeddings\n" );
    fprintf( stdout, "  wte: %7.2lf  %7.2lf \n", 
             m->wte[0], m->wte[vocab_size*H-1] );
    fprintf( stdout, "  wpe: %7.2lf  %7.2lf \n", 
             m->wpe[0], m->wpe[n_ctx*H-1] );

    for (int l=0; l < L; l++) {
        ao_layer_t* lp = &( m->layers[l] );
        fprintf( stdout, " Layer: %d\n", l );

        fprintf( stdout, "  LN1 epsilon: %g \n", lp->ln1.epsilon );
        fprintf( stdout, "  LN1 gamma: %7.2lf  %7.2lf \n", 
                 lp->ln1.gamma[0], lp->ln1.gamma[H-1] );
        fprintf( stdout, "  LN1  beta: %7.2lf  %7.2lf \n", 
                 lp->ln1.beta[0], lp->ln1.beta[H-1] );

        fprintf( stdout, "  Attn Wq,bq: %7.2lf  %7.2lf    %7.2lf  %7.2lf \n", 
                 lp->attn.Wq[0], lp->attn.Wq[H*H-1],
                 lp->attn.bq[0], lp->attn.bq[H-1] );
        fprintf( stdout, "  Attn Wk,bk: %7.2lf  %7.2lf    %7.2lf  %7.2lf \n", 
                 lp->attn.Wk[0], lp->attn.Wk[H*H-1],
                 lp->attn.bk[0], lp->attn.bk[H-1] );
        fprintf( stdout, "  Attn Wv,bv: %7.2lf  %7.2lf    %7.2lf  %7.2lf \n", 
                 lp->attn.Wv[0], lp->attn.Wv[H*H-1],
                 lp->attn.bv[0], lp->attn.bv[H-1] );
        fprintf( stdout, "  Attn Wo,bo: %7.2lf  %7.2lf    %7.2lf  %7.2lf \n", 
                 lp->attn.Wo[0], lp->attn.Wo[H*H-1],
                 lp->attn.bo[0], lp->attn.bo[H-1] );

        fprintf( stdout, "  LN2 epsilon: %g \n", lp->ln2.epsilon );
        fprintf( stdout, "  LN2 gamma: %7.2lf  %7.2lf \n", 
                 lp->ln2.gamma[0], lp->ln2.gamma[H-1] );
        fprintf( stdout, "  LN2  beta: %7.2lf  %7.2lf \n", 
                 lp->ln2.beta[0], lp->ln2.beta[H-1] );

        fprintf( stdout, "  FFN W1: %7.2lf  %7.2lf     %7.2lf  %7.2lf \n", 
                 lp->ffn.W1[0], lp->ffn.W1[F*H-1],
                 lp->ffn.b1[0], lp->ffn.b1[F-1] );
        fprintf( stdout, "  FFN W2: %7.2lf  %7.2lf     %7.2lf  %7.2lf \n", 
                 lp->ffn.W2[0], lp->ffn.W2[H*F-1],
                 lp->ffn.b2[0], lp->ffn.b2[H-1] );
    }

    fprintf( stdout, " Final LayerNorm\b\n" );
    fprintf( stdout, "  LNf epsilon: %g \n", m->ln_f.epsilon );
    fprintf( stdout, "  LNf gamma: %7.2lf  %7.2lf \n", 
                 m->ln_f.gamma[0], m->ln_f.gamma[H-1] );
    fprintf( stdout, "  LNF  beta: %7.2lf  %7.2lf \n", 
                 m->ln_f.beta[0], m->ln_f.beta[H-1] );
    fprintf( stdout, " =========== END ========== \n" );
}
