#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef int32_t token_id;

struct tokenizer;

/* lifecycle */

struct tokenizer *tokenizer_create(const char *model_path);
void tokenizer_destroy(struct tokenizer *t);

/* encode */

int tokenizer_encode(
    struct tokenizer *t,
    const char *text,
    token_id *tokens,
    int max_tokens
);

/* count tokens */

int tokenizer_count(
    struct tokenizer *t,
    const char *text
);

#endif

