#include <stdlib.h>
#include <string.h>

#include "inference.h"

struct inference_engine {
    struct tokenizer *tokenizer;
};

struct inference_engine *inference_create(struct tokenizer *tokenizer)
{
    struct inference_engine *e;

    e = malloc(sizeof(*e));
    if (!e)
        return NULL;

    e->tokenizer = tokenizer;

    return e;
}

void inference_destroy(struct inference_engine *e)
{
    free(e);
}

int inference_generate(
    struct inference_engine *e,
    struct session *s,
    const char *prompt,
    token_callback cb,
    void *user
)
{
    (void)e;
    (void)prompt;

    const char *parts[] = {
        "This ",
        "is ",
        "a ",
        "shim."
    };

    int n = sizeof(parts) / sizeof(parts[0]);

    for (int i = 0; i < n; i++) {

        int tok = tokenizer_count(e->tokenizer, parts[i]);

        if (tok > 0)
            s->generated_token_count += tok;

        s->token_count =
            s->prompt_token_count +
            s->generated_token_count;

        if (cb(parts[i], 0, user) != 0)
            return -1;
    }

    cb("", 1, user);

    return 0;
}


int inference_update_session_tokens(
    struct inference_engine *e,
    struct session *s
)
{
    int total = 0;

    if (!e || !s)
        return -1;

    for (int i = 0; i < s->message_count; i++) {

        struct message *m = &s->messages[i];

        int n = tokenizer_count(e->tokenizer, m->content);

        if (n < 0)
            return -1;

        m->token_count = n;

        total += n;
    }

    s->token_count = total;
    s->prompt_token_count = total;

    return total;
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

