#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "inference.h"
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
    void *ctx
)
{
    struct tokenizer *t = e->tokenizer;

    int vocab_size = tokenizer_vocab_size(t);
    token_id eos_id = tokenizer_eos_id(t);
    token_id bos_id = tokenizer_bos_id(t);
    token_id unk_id = tokenizer_unk_id(t);

    int generated = 0;

    (void)prompt;

//Randomize on the fly
//srand(time(NULL));

    while (1) {

        token_id id;

        if (generated >= 5) {
            id = eos_id;
        } else {

            /* draw random token */

            while (1) {

                id = rand() % vocab_size;

                if (id == bos_id)
                    continue;

                if (id == unk_id)
                    continue;

                break;
            }
        }

        /* decode token */

        char piece[64];

        int len = tokenizer_decode(t, id, piece, sizeof(piece));
        if (len < 0)
            return -1;

        /* stream piece */

        cb(piece, 0, ctx);

        /* update session token accounting */

        s->token_count++;
        s->generated_token_count++;

        generated++;

        if (id == eos_id) {
            cb("", 1, ctx);
            break;
        }
    }

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

