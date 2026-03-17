#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "transformer.h"

struct block {
    float *wq;
    float *wk;
    float *wv;
    float *wo;

    float *w1;
    float *w2;
};

struct transformer {
    struct model_config cfg;

    float *token_embedding;   /* vocab_size x hidden_size */
    float *lm_head;           /* hidden_size x vocab_size */

    struct block *blocks;

    /* scratch buffers */
    float *x;         /* context_length x hidden_size */
    float *attn;      /* context_length x hidden_size */
    float *ff;        /* context_length x hidden_size */
    float *k, *q, *v; /* context_length x hidden_size each */
    float *score;     /* context_length */
    float *tmp;       /* hidden_size */
    float *ff1;       /* expansion to ff_size */
    float *ff2;       /* contraction to hidden_size */
};

int transformer_get_context_length(struct transformer *t)
{
    if(!t)
        return -1;

    return t->cfg.context_length;
}

static void fill_rand(float *p, int n)
{
    for (int i = 0; i < n; i++) {
        p[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    }
}

struct transformer *transformer_create(const struct model_config *cfg)
{
    struct transformer *t;

    if (!cfg)
        return NULL;

    t = calloc(1, sizeof(*t));
    if (!t)
        return NULL;

    t->cfg = *cfg;

    int V = cfg->vocab_size;
    int H = cfg->hidden_size;
    int L = cfg->n_layers;
    int C = cfg->context_length;
    int F = cfg->ff_size;
    fprintf( stdout, " [DEBUG]  Transformer model architecture\n" );
    fprintf( stdout, "   vocab size: %d \n", V );
    fprintf( stdout, "   hidden dimension: %d floats (model dim) \n", H );
    fprintf( stdout, "   head size: %d \n", cfg->head_size );
    fprintf( stdout, "   number of layers: %d \n", L );
    fprintf( stdout, "   context length: %d tokens \n", C );
    fprintf( stdout, "   feed-forward size: %d \n", F );

    t->token_embedding = malloc(sizeof(float) * V * H);
    t->lm_head         = malloc(sizeof(float) * H * V);
    t->blocks          = calloc(L, sizeof(struct block));

    t->x    = malloc(sizeof(float) * C * H);
    t->attn = malloc(sizeof(float) * C * H);
    t->ff   = malloc(sizeof(float) * C * H);

    t->k    = malloc(sizeof(float) * C * H);
    t->q    = malloc(sizeof(float) * C * H);
    t->v    = malloc(sizeof(float) * C * H);
    t->score = malloc(sizeof(float) * C);
    t->tmp   = malloc(sizeof(float) * H);
    t->ff1   = malloc(sizeof(float) * F);
    t->ff2   = malloc(sizeof(float) * H);

    if (!t->token_embedding || !t->lm_head || !t->blocks ||
        !t->x || !t->attn || !t->ff) {
        transformer_destroy(t);
        return NULL;
    }

    fill_rand(t->token_embedding, V * H);
    fill_rand(t->lm_head, H * V);

    for (int i = 0; i < L; i++) {
        t->blocks[i].wq = malloc(sizeof(float) * H * H);
        t->blocks[i].wk = malloc(sizeof(float) * H * H);
        t->blocks[i].wv = malloc(sizeof(float) * H * H);
        t->blocks[i].wo = malloc(sizeof(float) * H * H);

        t->blocks[i].w1 = malloc(sizeof(float) * H * F);
        t->blocks[i].w2 = malloc(sizeof(float) * F * H);

        if (!t->blocks[i].wq || !t->blocks[i].wk || !t->blocks[i].wv ||
            !t->blocks[i].wo || !t->blocks[i].w1 || !t->blocks[i].w2) {
            transformer_destroy(t);
            return NULL;
        }

        fill_rand(t->blocks[i].wq, H * H);
        fill_rand(t->blocks[i].wk, H * H);
        fill_rand(t->blocks[i].wv, H * H);
        fill_rand(t->blocks[i].wo, H * H);
        fill_rand(t->blocks[i].w1, H * F);
        fill_rand(t->blocks[i].w2, F * H);
    }

    return t;
}

void transformer_destroy(struct transformer *t)
{
    if (!t)
        return;

    if (t->blocks) {
        for (int i = 0; i < t->cfg.n_layers; i++) {
            free(t->blocks[i].wq);
            free(t->blocks[i].wk);
            free(t->blocks[i].wv);
            free(t->blocks[i].wo);
            free(t->blocks[i].w1);
            free(t->blocks[i].w2);
        }
    }

    free(t->blocks);
    free(t->token_embedding);
    free(t->lm_head);
    free(t->x);
    free(t->attn);
    free(t->ff);
    free(t->k);
    free(t->q);
    free(t->v);
    free(t->score);
    free(t->tmp);
    free(t->ff1);
    free(t->ff2);
    free(t);
}

static void matvec(
    const float *x,
    const float *W,
    float *y,
    int in_dim,
    int out_dim
)
{
    for (int j = 0; j < out_dim; j++) {
        float s = 0.0f;

        for (int i = 0; i < in_dim; i++) {
            s += x[i] * W[i * out_dim + j];
        }

        y[j] = s;
    }
}

static float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

static float dot_product(const float *a, const float *b, int n)
{
    float s = 0.0f;

    for (int i = 0; i < n; i++)
        s += a[i] * b[i];

    return s;
}

static void add_inplace(float *dst, const float *src, int n)
{
    for (int i = 0; i < n; i++)
        dst[i] += src[i];
}

static void zero_vec(float *x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = 0.0f;
}

static void softmax_inplace(float *x, int n)
{
    float maxv = x[0];

    for (int i = 1; i < n; i++) {
        if (x[i] > maxv)
            maxv = x[i];
    }

    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }

    if (sum == 0.0f)
        return;

    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

static void attention_forward(
    struct transformer *t,
    struct block *b,
    int n_tokens
)
{
    int H = t->cfg.hidden_size;
    float scale = 1.0f / sqrtf((float)H);

    /* 1. Compute Q, K, V for all positions */
    for (int pos = 0; pos < n_tokens; pos++) {
        float *xrow = &t->x[pos * H];
        float *qrow = &t->q[pos * H];
        float *krow = &t->k[pos * H];
        float *vrow = &t->v[pos * H];

        matvec(xrow, b->wq, qrow, H, H);
        matvec(xrow, b->wk, krow, H, H);
        matvec(xrow, b->wv, vrow, H, H);
    }

    /* 2. Causal attention for each position */
    for (int tpos = 0; tpos < n_tokens; tpos++) {
        float *qrow = &t->q[tpos * H];
        float *out  = &t->attn[tpos * H];

        zero_vec(out, H);

        for (int j = 0; j <= tpos; j++) {
            float *krow = &t->k[j * H];
            t->score[j] = dot_product(qrow, krow, H) * scale;
        }

        softmax_inplace(t->score, tpos + 1);

        for (int j = 0; j <= tpos; j++) {
            float *vrow = &t->v[j * H];
            float a = t->score[j];

            for (int h = 0; h < H; h++)
                out[h] += a * vrow[h];
        }
    }

    /* 3. Output projection + residual add into x */
    for (int pos = 0; pos < n_tokens; pos++) {
        float *attn_row = &t->attn[pos * H];
        float *xrow     = &t->x[pos * H];

        matvec(attn_row, b->wo, t->tmp, H, H);
        add_inplace(xrow, t->tmp, H);
    }
}


void block_forward(
    struct transformer *t,
    struct block *b,
    int n_tokens
)
{
#ifdef _DEBUG3_
    fprintf( stdout, " [DEBUG]  Transformer block forward (all tokens)\n" );
#endif
    int H = t->cfg.hidden_size;
    int F = t->cfg.ff_size;

    attention_forward(t, b, n_tokens);

    /* feed-forward per token */
    for (int pos = 0; pos < n_tokens; pos++) {
        float *xrow = &t->x[pos * H];

        matvec(xrow, b->w1, t->ff1, H, F);

        for (int i = 0; i < F; i++) {
            if (t->ff1[i] < 0.0f)
                t->ff1[i] = 0.0f;
        }

        matvec(t->ff1, b->w2, t->ff2, F, H);
        add_inplace(xrow, t->ff2, H);
    }
}

int transformer_forward(
    struct transformer *t,
    const token_id *tokens,
    int n_tokens,
    float *logits
)
{
    if (!t || !tokens || !logits)
        return -1;

    int V = t->cfg.vocab_size;
    int H = t->cfg.hidden_size;

    if (n_tokens <= 0 || n_tokens > t->cfg.context_length)
        return -1;

    /* embedding lookup */
    for (int pos = 0; pos < n_tokens; pos++) {
        token_id tok = tokens[pos];

        if (tok < 0 || tok >= V)
            return -1;

        memcpy(
            &t->x[pos * H],
            &t->token_embedding[tok * H],
            sizeof(float) * H
        );
    }

    float *tmp1 = malloc(sizeof(float) * H);
    float *tmp2 = malloc(sizeof(float) * H);

    if (!tmp1 || !tmp2) {
        free(tmp1);
        free(tmp2);
        return -1;
    }

    for (int l = 0; l < t->cfg.n_layers; l++) {
#ifdef _DEBUG_
       fprintf( stdout, " [DEBUG]  Transformer forward (layer: %d)\n", l );
#endif
        for (int pos = 0; pos < n_tokens; pos++) {
            block_forward(
                t,
                &t->blocks[l],
                n_tokens
            );
        }
    }

    /* logits from last token */
    float *last = &t->x[(n_tokens - 1) * H];

    for (int v = 0; v < V; v++) {
        float s = 0.0f;

        for (int i = 0; i < H; i++) {
            s += last[i] * t->lm_head[i * V + v];
        }

        logits[v] = s;
    }

    free(tmp1);
    free(tmp2);

    return 0;
}

