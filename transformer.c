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

void block_forward(
    struct transformer *t,
    struct block *b,
    float *x_row,
    float *tmp1,
    float *tmp2
)
{
#ifdef _DEBUG3_
    fprintf( stdout, " [DEBUG]  Transformer forward\n" );
#endif
    int H = t->cfg.hidden_size;
    int F = t->cfg.ff_size;

    /* placeholder "attention-like" transform */
    matvec(x_row, b->wq, tmp1, H, H);
    matvec(tmp1, b->wo, tmp2, H, H);

    for (int i = 0; i < H; i++) {
        x_row[i] += tmp2[i];
    }

    /* feed-forward */
    float *ff1 = malloc(sizeof(float) * F);
    float *ff2 = malloc(sizeof(float) * H);

    if (!ff1 || !ff2) {
        free(ff1);
        free(ff2);
        return;
    }

    matvec(x_row, b->w1, ff1, H, F);

    for (int i = 0; i < F; i++) {
        ff1[i] = relu(ff1[i]);
    }

    matvec(ff1, b->w2, ff2, F, H);

    for (int i = 0; i < H; i++) {
        x_row[i] += ff2[i];
    }

    free(ff1);
    free(ff2);
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
        for (int pos = 0; pos < n_tokens; pos++) {
            block_forward(
                t,
                &t->blocks[l],
                &t->x[pos * H],
                tmp1,
                tmp2
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

