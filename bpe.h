#ifndef BPE_H
#define BPE_H

#include "vocab.h"

struct merge_entry {
    token_id left;
    token_id right;
    int rank;
    token_id result;
};

struct bpe;

/* lifecycle */

struct bpe *bpe_create(int merge_count);
void bpe_destroy(struct bpe *b);

/* insertion */

int bpe_insert(
    struct bpe *b,
    token_id left,
    token_id right,
    int rank,
    token_id result
);

/* lookup */

const struct merge_entry *bpe_lookup(
    struct bpe *b,
    token_id left,
    token_id right
);

#ifdef _DEBUG_
void bpe_display( struct bpe *b );
#endif

#endif

