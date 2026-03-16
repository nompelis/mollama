#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "bpe.h"

struct bpe {
    int merge_count;
    int inserted_count;

    struct merge_entry *entries;

    int hash_cap;
    int *hash_slots;   /* stores index into entries, -1 means empty */
};

static uint32_t hash_pair(token_id left, token_id right)
{
    uint32_t h = 2166136261u;

    h ^= (uint32_t)left;
    h *= 16777619u;

    h ^= (uint32_t)right;
    h *= 16777619u;

    return h;
}

static int next_pow2(int x)
{
    int p = 1;

    while (p < x)
        p <<= 1;

    return p;
}

struct bpe *bpe_create(int merge_count)
{
    struct bpe *b;

    if (merge_count < 0)
        return NULL;

    b = malloc(sizeof(*b));
    if (!b)
        return NULL;

    b->merge_count = merge_count;
    b->inserted_count = 0;

    b->entries = NULL;
    b->hash_slots = NULL;
    b->hash_cap = 0;

    if (merge_count == 0)
        return b;

    b->entries = calloc((size_t)merge_count, sizeof(struct merge_entry));
    if (!b->entries) {
        free(b);
        return NULL;
    }

    b->hash_cap = next_pow2(merge_count * 2);
    b->hash_slots = malloc((size_t)b->hash_cap * sizeof(int));
    if (!b->hash_slots) {
        free(b->entries);
        free(b);
        return NULL;
    }

    for (int i = 0; i < b->hash_cap; i++)
        b->hash_slots[i] = -1;

    return b;
}

#ifdef _DEBUG_
void bpe_display( struct bpe *b )
{
    fprintf( stdout, " [DEBUG]  BPE contents:\n" );
    for (int idx = 0; idx < b->inserted_count;++idx) {
        fprintf( stdout, " (%d)  left: %d, right: %d, rank: %d, result: %d\n",
        idx,
        b->entries[idx].left,
        b->entries[idx].right,
        b->entries[idx].rank,
        b->entries[idx].result );
    }
}
#endif

void bpe_destroy(struct bpe *b)
{
    if (!b)
        return;

    free(b->entries);
    free(b->hash_slots);
    free(b);
}

int bpe_insert(
    struct bpe *b,
    token_id left,
    token_id right,
    int rank,
    token_id result
)
{
    if (!b)
        return -1;

    if (b->inserted_count >= b->merge_count)
        return -1;

    int idx = b->inserted_count++;

    b->entries[idx].left = left;
    b->entries[idx].right = right;
    b->entries[idx].rank = rank;
    b->entries[idx].result = result;

    if (b->hash_cap == 0)
        return 0;

    uint32_t h = hash_pair(left, right);
    int slot = (int)(h & (uint32_t)(b->hash_cap - 1));

    while (b->hash_slots[slot] != -1) {
        slot = (slot + 1) & (b->hash_cap - 1);
    }

    b->hash_slots[slot] = idx;

    return 0;
}

const struct merge_entry *bpe_lookup(
    struct bpe *b,
    token_id left,
    token_id right
)
{
    if (!b || b->hash_cap == 0)
        return NULL;

#ifdef _DEBUG3_
    fprintf( stdout, " BPE trial pair: %d,%d\n", left, right );
#endif
    uint32_t h = hash_pair(left, right);
    int slot = (int)(h & (uint32_t)(b->hash_cap - 1));

    while (1) {
        int idx = b->hash_slots[slot];
#ifdef _DEBUG3_
        fprintf( stdout, " BPE hash_slots idx: %d\n", idx );
#endif

        if (idx == -1)
            return NULL;

        const struct merge_entry *e = &b->entries[idx];

        if (e->left == left && e->right == right)
            return e;

        slot = (slot + 1) & (b->hash_cap - 1);
    }
}

