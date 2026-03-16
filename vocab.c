#include <stdlib.h>
#include <string.h>

#include "vocab.h"

struct vocab {
    int vocab_size;

    struct vocab_entry *entries;

    /* hash table */

    int hash_cap;
    token_id *hash_ids;
};


static uint32_t hash_bytes(const char *s, int len)
{
    uint32_t h = 2166136261u;

    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619;
    }

    return h;
}

static int next_pow2(int x)
{
    int p = 1;

    while (p < x)
        p <<= 1;

    return p;
}

struct vocab *vocab_create(int vocab_size)
{
    struct vocab *v;

    v = malloc(sizeof(*v));
    if (!v)
        return NULL;

    v->vocab_size = vocab_size;

    v->entries = calloc(vocab_size, sizeof(struct vocab_entry));
    if (!v->entries) {
        free(v);
        return NULL;
    }

    v->hash_cap = next_pow2(vocab_size * 2);

    v->hash_ids = malloc(sizeof(token_id) * v->hash_cap);
    if (!v->hash_ids) {
        free(v->entries);
        free(v);
        return NULL;
    }

    for (int i = 0; i < v->hash_cap; i++)
        v->hash_ids[i] = -1;

    return v;
}

void vocab_destroy(struct vocab *v)
{
    if (!v)
        return;

    for (int i = 0; i < v->vocab_size; i++) {
        free(v->entries[i].piece);
    }

    free(v->entries);
    free(v->hash_ids);
    free(v);
}

int vocab_insert(
    struct vocab *v,
    token_id id,
    const char *piece,
    int piece_len
)
{
    if (!v || id < 0 || id >= v->vocab_size)
        return -1;

    char *p = malloc(piece_len + 1);
    if (!p)
        return -1;

    memcpy(p, piece, piece_len);
    p[piece_len] = '\0';

    v->entries[id].piece = p;
    v->entries[id].piece_len = piece_len;

    uint32_t h = hash_bytes(piece, piece_len);
    int slot = h & (v->hash_cap - 1);

    while (v->hash_ids[slot] != -1)
        slot = (slot + 1) & (v->hash_cap - 1);

    v->hash_ids[slot] = id;

    return 0;
}

token_id vocab_lookup_piece(
    struct vocab *v,
    const char *piece,
    int piece_len
)
{
    if (!v)
        return -1;

    uint32_t h = hash_bytes(piece, piece_len);
    int slot = h & (v->hash_cap - 1);

    while (1) {

        token_id id = v->hash_ids[slot];

        if (id == -1)
            return -1;

        struct vocab_entry *e = &v->entries[id];

        if (e->piece_len == piece_len &&
            memcmp(e->piece, piece, piece_len) == 0)
            return id;

        slot = (slot + 1) & (v->hash_cap - 1);
    }
}

const struct vocab_entry *
vocab_lookup_id(
    struct vocab *v,
    token_id id
)
{
    if (!v)
        return NULL;

    if (id < 0 || id >= v->vocab_size)
        return NULL;

    return &v->entries[id];
}

