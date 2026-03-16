#ifndef SAMPLER_H
#define SAMPLER_H

#include "vocab.h"

struct sampler;

/* lifecycle */

struct sampler *sampler_create(void);
void sampler_destroy(struct sampler *s);

/* sampling */

token_id sampler_random(
    struct sampler *s,
    int vocab_size,
    token_id bos,
    token_id eos,
    token_id unk
);

token_id sampler_sample(
    struct sampler *s,
    const float *logits,
    int vocab_size,
    token_id bos,
    token_id eos,
    token_id unk
);

#endif

