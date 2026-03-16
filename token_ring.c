#include <stdlib.h>

#include "token_ring.h"

struct token_ring *token_ring_create(int capacity)
{
    struct token_ring *r;

    r = calloc(1, sizeof(*r));
    if (!r)
        return NULL;

    r->data = malloc(sizeof(token_id) * capacity);
    if (!r->data) {
        free(r);
        return NULL;
    }

    r->capacity = capacity;
    r->start = 0;
    r->count = 0;

    return r;
}

void token_ring_destroy(struct token_ring *r)
{
    if (!r)
        return;

    free(r->data);
    free(r);
}

int token_ring_push(struct token_ring *r, token_id tok)
{
    if (!r)
        return -1;

    int pos;

    if (r->count < r->capacity) {

        pos = (r->start + r->count) % r->capacity;
        r->data[pos] = tok;
        r->count++;

    } else {

        /* overwrite oldest */

        pos = r->start;
        r->data[pos] = tok;

        r->start = (r->start + 1) % r->capacity;
    }

    return 0;
}

int token_ring_get(
    struct token_ring *r,
    int index,
    token_id *out
)
{
    if (!r || !out)
        return -1;

    if (index < 0 || index >= r->count)
        return -1;

    int pos = (r->start + index) % r->capacity;

    *out = r->data[pos];

    return 0;
}

int token_ring_size(struct token_ring *r)
{
    if (!r)
        return 0;

    return r->count;
}

int token_ring_last(struct token_ring *r, token_id *out)
{
    if (!r || !out)
        return -1;

    if (r->count == 0)
        return -1;

    int pos = (r->start + r->count - 1) % r->capacity;

    *out = r->data[pos];

    return 0;
}

