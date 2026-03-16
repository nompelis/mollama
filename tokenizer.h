#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include "vocab.h"


/* lifecycle */

struct tokenizer *tokenizer_create(const char *model_path);
void tokenizer_destroy(struct tokenizer *t);

/* accessors */

int tokenizer_vocab_size(struct tokenizer *t);

token_id tokenizer_bos_id(struct tokenizer *t);
token_id tokenizer_eos_id(struct tokenizer *t);
token_id tokenizer_unk_id(struct tokenizer *t);

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

/* dencode */

const char *tokenizer_decode_piece(
    struct tokenizer *t,
    token_id tok
);

void tokenizer_debug_tokens(
    struct tokenizer *t,
    token_id *tokens,
    int count
);

int tokenizer_decode(
    struct tokenizer *t,
    token_id tok,
    char *out,
    int maxlen
);

#ifdef _DEBUG_
void tokenizer_vocab_display(struct tokenizer *t, int count);
#endif

#endif

