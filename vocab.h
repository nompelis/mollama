#ifndef VOCAB_H
#define VOCAB_H

#include <stddef.h>
#include <stdint.h>

typedef int32_t token_id;

struct vocab_entry {
    char *piece;
    int piece_len;
};

struct vocab;

/* lifecycle */

struct vocab *vocab_create(int vocab_size);
void vocab_destroy(struct vocab *v);

/* insertion (during tokenizer initialization) */

int vocab_insert(
    struct vocab *v,
    token_id id,
    const char *piece,
    int piece_len
);

/* lookup */

token_id vocab_lookup_piece(
    struct vocab *v,
    const char *piece,
    int piece_len
);

const struct vocab_entry *
vocab_lookup_id(
    struct vocab *v,
    token_id id
);

#endif

