#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "vocab.h"

struct model_config {
    int vocab_size;
    int context_length;

    int n_layers;
    int n_heads;

    int hidden_size;
    int head_size;
    int ff_size;
};

struct transformer;

struct transformer *transformer_create(const struct model_config *cfg);
void transformer_destroy(struct transformer *t);

/* forward pass: fills logits for next-token prediction */
int transformer_forward(
    struct transformer *t,
    const token_id *tokens,
    int n_tokens,
    float *logits
);

/* prefill and step */
int transformer_prefill(
    struct transformer *t,
    const token_id *tokens,
    int n_tokens,
    float *logits
);

int transformer_step(
    struct transformer *t,
    token_id tok,
    float *logits
);

int transformer_get_context_length(struct transformer *t);

#endif

