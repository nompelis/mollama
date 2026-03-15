#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "tokenizer.h"

struct tokenizer {
    int dummy;
};

struct tokenizer *tokenizer_create(const char *model_path)
{
    struct tokenizer *t = malloc(sizeof(*t));

    if (!t)
        return NULL;

    t->dummy = 0;

    return t;
}

void tokenizer_destroy(struct tokenizer *t)
{
    free(t);
}

static int is_delim(char c)
{
    return isspace((unsigned char)c);
}

int tokenizer_encode(
    struct tokenizer *t,
    const char *text,
    token_id *tokens,
    int max_tokens
)
{
    int count = 0;

    const char *p = text;

    while (*p) {

        while (*p && is_delim(*p))
            p++;

        if (!*p)
            break;

        if (count >= max_tokens)
            return -1;

        tokens[count++] = count;

        while (*p && !is_delim(*p))
            p++;
    }

    return count;
}

int tokenizer_count(
    struct tokenizer *t,
    const char *text
)
{
    token_id tmp[4096];

    return tokenizer_encode(t, text, tmp, 4096);
}

