#ifndef TOKEN_RING_H
#define TOKEN_RING_H

#include "vocab.h"

struct token_ring {
    token_id *data;
    int capacity;

    int start;   /* oldest token */
    int count;   /* number of tokens stored */
};

/* lifecycle */

struct token_ring *token_ring_create(int capacity);
void token_ring_destroy(struct token_ring *r);

/* operations */

int token_ring_push(struct token_ring *r, token_id tok);

int token_ring_get(
    struct token_ring *r,
    int index,
    token_id *out
);

int token_ring_size(struct token_ring *r);

int token_ring_last(struct token_ring *r, token_id *out);

#endif
