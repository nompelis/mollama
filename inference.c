#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "inference.h"
#include "token_ring.h"
#include "sampler.h"

struct inference_engine {
    struct tokenizer *tokenizer;
    struct transformer *transformer;
};

struct inference_ctx {
    struct tokenizer *tok;
    struct token_ring *ring;
    struct sampler *sampler;

    struct session *session;

    token_callback callback;
    void *cb_ctx;

    token_id *tokens;
    float *logits;
    int context_length;
    int generated;
};


struct inference_engine *inference_create(struct tokenizer *tokenizer,
                                          struct transformer *transformer)
{
    struct inference_engine *e;

    e = malloc(sizeof(*e));
    if (!e)
        return NULL;

    e->tokenizer = tokenizer;
    e->transformer = transformer;

    return e;
}

void inference_destroy(struct inference_engine *e)
{
    free(e);
}

static int load_prompt_into_ring(
    struct tokenizer *tok,
    struct token_ring *ring,
    const char *prompt
)
{
    token_id buf[512];

    int n = tokenizer_encode(tok, prompt, buf, 512);
    if (n < 0)
        return -1;

    for (int i = 0; i < n; i++)
        token_ring_push(ring, buf[i]);

    return n;
}

int inference_generate(
    struct inference_engine *e,
    struct session *s,
    const char *prompt,
    token_callback cb,
    void *cb_ctx
)
{
    struct tokenizer *t = e->tokenizer;

    int vocab_size = tokenizer_vocab_size(t);
    token_id eos_id = tokenizer_eos_id(t);
    token_id bos_id = tokenizer_bos_id(t);
    token_id unk_id = tokenizer_unk_id(t);

    int context_length = transformer_get_context_length(e->transformer);

    struct token_ring *ring = token_ring_create(context_length);
    if (!ring)
        return -1;

    struct sampler *sampler = sampler_create();
    if (!sampler)
        return -1;

    load_prompt_into_ring(t, ring, prompt);

    // Items the inference needs
    struct inference_ctx ctx = { .tok=t, .ring=ring, .sampler=sampler,
                                 .session=s,
                                 .callback=cb, .cb_ctx=cb_ctx,
                                 .tokens=NULL, .logits=NULL,
                                 .context_length=context_length, .generated=0 };

    // extracting tokens from ring buffer to a contiguous array
    ctx.tokens = (token_id*) malloc(sizeof(token_id) * ctx.context_length);
    if (!ctx.tokens)
        return -1;

    int ring_size = token_ring_size(ring);
    for (int i = 0; i < ring_size; ++i)
        token_ring_get(ring, i, &(ctx.tokens[i]));

    ctx.logits = (float*) malloc(sizeof(float) * vocab_size);
    if (!ctx.logits)
        return -1;

    (void)prompt;

    while (1) {

        token_id id;

        if (ctx.generated >= 5) {
            id = eos_id;
        } else {

            /* draw random token */
            /* uses the sampler, like real inference would */

            for (int i = 0; i < vocab_size; i++)
                ctx.logits[i] = (float)rand();

            /* over-write with real inference */
            transformer_forward(e->transformer,
                                ctx.tokens, ring_size, ctx.logits);

            id = sampler_sample(ctx.sampler, ctx.logits,
                                vocab_size, bos_id, eos_id, unk_id);
        }

        /* push token to context */

        token_ring_push(ctx.ring, id);

        /* decode token */

        char piece[256];

        int len = tokenizer_decode(t, id, piece, sizeof(piece));
        if (len < 0)
            return -1;

        /* stream piece */

        cb(piece, 0, cb_ctx);

        /* update session token accounting */

        s->token_count++;
        s->generated_token_count++;

        ctx.generated++;

        if (id == eos_id) {
            cb("", 1, cb_ctx);
            break;
        }
    }

    free(ctx.tokens);
    free(ctx.logits);
    sampler_destroy(sampler);
    token_ring_destroy(ring);

    return 0;
}

int inference_update_prompt_tokens(
    struct inference_engine *e,
    struct session *s,
    const char *prompt
)
{
    if (!e || !s || !prompt)
        return -1;

    for (int i = 0; i < s->message_count; i++) {
        struct message *m = &s->messages[i];

        int n = tokenizer_count(e->tokenizer, m->content);
        if (n < 0)
            return -1;

        m->token_count = n;
    }

    int prompt_tokens = tokenizer_count(e->tokenizer, prompt);
    if (prompt_tokens < 0)
        return -1;

    s->prompt_token_count = prompt_tokens;
    s->token_count = prompt_tokens;

    return prompt_tokens;
}

