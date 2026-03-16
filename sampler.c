#include <stdlib.h>
#include <float.h>
#include <time.h>

#include "sampler.h"

struct sampler {
    unsigned int seed;
};

struct sampler *sampler_create(void)
{
    struct sampler *s;

    s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;

    s->seed = (unsigned int)time(NULL);

    srand(s->seed);

    return s;
}

void sampler_destroy(struct sampler *s)
{
    if (!s)
        return;

    free(s);
}

token_id sampler_random(
    struct sampler *s,
    int vocab_size,
    token_id bos,
    token_id eos,
    token_id unk
)
{
    token_id id;

    (void)s;

    while (1) {

        id = rand() % vocab_size;

        if (id == bos)
            continue;

        if (id == unk)
            continue;

        break;
    }

    return id;
}


token_id sampler_sample(
    struct sampler *s,
    const float *logits,
    int vocab_size,
    token_id bos,
    token_id eos,
    token_id unk
)
{
    token_id best = -1;
    float best_val = -FLT_MAX;

    (void)s;

    for (int i = 0; i < vocab_size; i++) {

        if (i == bos || i == unk)
            continue;

        float v = logits[i];

        if (v > best_val) {
            best_val = v;
            best = i;
        }
    }

    if (best < 0)
        best = eos;

    return best;
}

