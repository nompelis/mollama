#ifndef INFERENCE_H
#define INFERENCE_H

#include "tokenizer.h"
#include "session.h"

struct inference_engine;

/* lifecycle */

struct inference_engine *
inference_create(struct tokenizer *tokenizer);

void inference_destroy(struct inference_engine *e);

/* generation callback */

typedef int (*token_callback)(
    const char *text_fragment,
    int done,
    void *user
);

/* generate */

int inference_generate(
    struct inference_engine *e,
    struct session *s,
    const char *prompt,
    token_callback cb,
    void *user
);

int inference_update_session_tokens(
    struct inference_engine *e,
    struct session *s
);

int inference_update_prompt_tokens(
    struct inference_engine *e,
    struct session *s,
    const char *prompt
);

#endif
