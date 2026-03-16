#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "tokenizer.h"
#include "bpe.h"


struct tokenizer {
    int vocab_size;
    struct vocab *vocab;

    struct bpe *bpe;

    int merge_count;
    struct merge_entry *merges;

//  /* fast lookup tables */
//  int *piece_hash_slots;
//  int piece_hash_cap;

//  int *merge_hash_slots;
//  int merge_hash_cap;

    /* special tokens */
    token_id bos_id;
    token_id eos_id;
    token_id unk_id;
};


int tokenizer_vocab_size(struct tokenizer *t)
{
    if (!t || !t->vocab)
        return -1;

    return t->vocab_size;
}

token_id tokenizer_bos_id(struct tokenizer *t)
{
    return t ? t->bos_id : -1;
}

token_id tokenizer_eos_id(struct tokenizer *t)
{
    return t ? t->eos_id : -1;
}

token_id tokenizer_unk_id(struct tokenizer *t)
{
    return t ? t->unk_id : -1;
}


void tokenizer_debug_tokens(
    struct tokenizer *t,
    token_id *tokens,
    int count
)
{
    if (!t || !tokens)
        return;

    for (int i = 0; i < count; i++) {

        token_id id = tokens[i];

        const struct vocab_entry *e =
            vocab_lookup_id(t->vocab, id);

        if (!e) {
            printf("%d: <invalid>\n", id);
            continue;
        }

        fprintf( stdout, "%d: \"", id);

        fwrite(e->piece, 1, e->piece_len, stdout);

        fprintf( stdout, "\"\n");
    }
}

#ifdef _DEBUG_
void tokenizer_vocab_display(struct tokenizer *t, int count)
{
   for( int id = 0; id < count; ++id) {
        const struct vocab_entry *e = vocab_lookup_id(t->vocab, id);

        if (!e) {
            printf("%d: <invalid>\n", id);
            continue;
        }

        fprintf( stdout, "%d: \"", id);

        fwrite(e->piece, 1, e->piece_len, stdout);

        fprintf( stdout, "\"\n");
   }
}
#endif

static int tokenizer_build_test_vocab(struct tokenizer *t)
{
    t->vocab_size = 32;
    t->vocab = vocab_create(32);
    if (!t->vocab)
        return -1;

    /* special tokens */

    vocab_insert(t->vocab, 0, "<unk>", 5);
    vocab_insert(t->vocab, 1, "<bos>", 5);
    vocab_insert(t->vocab, 2, "<eos>", 5);

    t->unk_id = 0;
    t->bos_id = 1;
    t->eos_id = 2;

    /* basic pieces */
// 0  <unk>
// 1  <bos>
// 2  <eos>
// 
// 3  "This"
// 4  "is"
// 5  "a"
// 6  "shim"
// 7  "."
// 8  " "
// 9  "Hello"
// 10 "world"

    vocab_insert(t->vocab, 3, "This", 4);
    vocab_insert(t->vocab, 4, "is", 2);
    vocab_insert(t->vocab, 5, "a", 1);
    vocab_insert(t->vocab, 6, "shim", 4);
    vocab_insert(t->vocab, 7, ".", 1);
    vocab_insert(t->vocab, 8, " ", 1);

    vocab_insert(t->vocab, 9, "Hello", 5);
    vocab_insert(t->vocab, 10, "world", 5);

    vocab_insert(t->vocab, 11, "<system>", 8);
    vocab_insert(t->vocab, 12, "</system>", 9);

    vocab_insert(t->vocab, 13, "<user>", 6);
    vocab_insert(t->vocab, 14, "</user>", 7);

    vocab_insert(t->vocab, 15, "<assistant>", 11);
    vocab_insert(t->vocab, 16, "</assistant>", 12);

    vocab_insert(t->vocab, 17, "\n", 1);

    return 0;
}

struct tokenizer *tokenizer_create(const char *model_path)
{
    struct tokenizer *t = malloc(sizeof(*t));

    if (!t) {
        fprintf( stdout, " [Error]  Could not create tokenizer\n" );
        return NULL;
    } else {
        fprintf( stdout, " [DEBUG]  Created tokenizer struct\n");
    }

    if (tokenizer_build_test_vocab(t) != 0) {
        fprintf( stdout, " [Error]  Could not create vocabulary\n" );
        free(t);
        return NULL;
    } else {
        fprintf( stdout, " [DEBUG]  Built vocabulary\n");
    }
#ifdef _DEBUG_
    tokenizer_vocab_display(t,16);
#endif

    // Make a BPE object

    t->bpe = bpe_create(4);
    if (!t->bpe) {
        fprintf( stdout, " [Error]  Could not create BPE struct\n" );
        tokenizer_destroy(t);
        return NULL;
    } else {
        fprintf( stdout, " [DEBUG]  Created BPE struct\n");
    }

    /* toy merges */
// ("shim","." ) → token 7
// ("Hello"," ") → token 9

    bpe_insert(t->bpe, 6, 7, 0, 6);  /* shim + . */
    bpe_insert(t->bpe, 9, 8, 1, 9);  /* Hello + space */

#ifdef _DEBUG_
    bpe_display(t->bpe);
#endif
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

static int bpe_apply_merges(
    struct tokenizer *t,
    token_id *syms,
    int count
)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Applying BPE merges\n");
#endif
    while (1) {

        int best_rank = -1;
        int best_pos  = -1;
        token_id best_tok = -1;

        for (int i = 0; i < count - 1; i++) {

            const struct merge_entry *m =
                bpe_lookup(t->bpe, syms[i], syms[i + 1]);

            if (!m)
                continue;

            if (best_rank == -1 || m->rank < best_rank) {
                best_rank = m->rank;
                best_pos  = i;
                best_tok  = m->result;
            }
  printf("i=%d, best_post: %d \n",i,best_pos);//HACK
        }

        if (best_pos < 0)
            break;

        syms[best_pos] = best_tok;

        for (int i = best_pos + 1; i < count - 1; i++)
            syms[i] = syms[i + 1];

        count--;
    }

#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Exiting apply-merges; count: %d\n", count );
#endif
    return count;
}

int tokenizer_encode(
    struct tokenizer *t,
    const char *text,
    token_id *tokens,
    int max_tokens
)
{
// scan text
// split into spaces, words (alpha), numbers (digit), punctuation
// lookup each piece in vocab
// emit token or <unk>

    if (!t || !text || !tokens)
        return -1;

#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Tokenizer encode (line: %d)\n", __LINE__ );
#endif
    token_id syms[256];
    int sym_count = 0;

    const char *p = text;

    while (*p) {
#ifdef _DEBUG2_
        fprintf( stdout, " Token: (position: %d) '%c'", (int) (p-text),
                 *p == '\n' ? '\\' : *p );
#endif

        /* spaces */

        if (isspace((unsigned char)*p)) {

            char buf[2] = { ' ', 0 };

            token_id id = vocab_lookup_piece(t->vocab, buf, 1);

            if (id < 0)
                id = t->unk_id;
#ifdef _DEBUG2_
            else {
                const struct vocab_entry *e = vocab_lookup_id(t->vocab, id);
                fprintf( stdout, " space -->%c<-- token id: %d (\"%s\",%d)\n",
                         *p, id, e->piece, e->piece_len );
            }
            if( id == t->unk_id ) fprintf( stdout, "\n");
#endif

            syms[sym_count++] = id;

            p++;
            continue;
        }

        /* letters */

        if (isalpha((unsigned char)*p)) {

            const char *start = p;

            while (isalpha((unsigned char)*p))
                p++;

            int len = (int)(p - start);

            token_id id = vocab_lookup_piece(t->vocab, start, len);

            if (id < 0)
                id = t->unk_id;
#ifdef _DEBUG2_
            else {
                char string[20];
                snprintf( string, sizeof(string), "%s", start );
                string[len] = '\0';
                const struct vocab_entry *e = vocab_lookup_id(t->vocab, id);
//if( e==NULL ) printf("SCREAM start=\"%s\" id=%d\n",start,id);//HACK
                fprintf( stdout, " alpha -->%s<-- token id: %d (\"%s\",%d)\n",
                         string, id, e->piece, e->piece_len );
            }
            if( id == t->unk_id ) fprintf( stdout, "\n");
#endif

            syms[sym_count++] = id;

            continue;
        }

        /* digits */

        if (isdigit((unsigned char)*p)) {

            const char *start = p;

            while (isdigit((unsigned char)*p))
                p++;

            int len = (int)(p - start);

            token_id id = vocab_lookup_piece(t->vocab, start, len);

            if (id < 0)
                id = t->unk_id;
#ifdef _DEBUG2_
            else {
                char string[20];
                snprintf( string, sizeof(string), "%s", start );
                string[len] = '\0';
                const struct vocab_entry *e = vocab_lookup_id(t->vocab, id);
                fprintf( stdout, " digit -->%s<-- token id: %d (\"%s\",%d)\n",
                         string, id, e->piece, e->piece_len );
            }
            if( id == t->unk_id ) fprintf( stdout, "\n");
#endif

            syms[sym_count++] = id;

            continue;
        }

        /* punctuation */

        if (ispunct((unsigned char)*p)) {

            char buf[2];
            buf[0] = *p;
            buf[1] = '\0';

            token_id id = vocab_lookup_piece(t->vocab, buf, 1);

            if (id < 0)
                id = t->unk_id;
#ifdef _DEBUG2_
            else {
                const struct vocab_entry *e = vocab_lookup_id(t->vocab, id);
                fprintf( stdout, " punct -->%c<-- token id: %d (\"%s\",%d)\n",
                         *p, id, e->piece, e->piece_len );
            }
            if( id == t->unk_id ) fprintf( stdout, "\n");
#endif

            syms[sym_count++] = id;

            p++;
            continue;
        }

        /* fallback */
#ifdef _DEBUG_
        fprintf( stdout, " [DEBUG]  Tokenizer encoder hit fallback \"%s\"\n",p);
#endif

        p++;
    }

    /* apply BPE merges */

    sym_count = bpe_apply_merges(t, syms, sym_count);

    if (sym_count > max_tokens)
        return -1;

    for (int i = 0; i < sym_count; i++)
        tokens[i] = syms[i];

#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Tokenizer worked on %d symbols\n", sym_count );
#endif
    return sym_count;
}

int tokenizer_count(
    struct tokenizer *t,
    const char *text
)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Tokenizer count \n" );
#endif
    token_id tmp[4096];

    return tokenizer_encode(t, text, tmp, 4096);
}

int tokenizer_decode(
    struct tokenizer *t,
    token_id tok,
    char *out,
    int maxlen
)
{
#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Tokenizer decode (line: %d)\n", __LINE__ );
#endif
    const struct vocab_entry *e;

    if (!t || !out)
        return -1;

    e = vocab_lookup_id(t->vocab, tok);
    if (!e)
        return -1;

    if (e->piece_len + 1 > maxlen)
        return -1;

    memcpy(out, e->piece, e->piece_len);
    out[e->piece_len] = '\0';
#ifdef _DEBUG_
    fprintf( stdout, " token id: %d, ASCII: \"%s\"\n", tok, out );
#endif

    return e->piece_len;
}

