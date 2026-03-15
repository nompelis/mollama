#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "session.h"

static struct session sessions[NS];
static pthread_mutex_t table_lock = PTHREAD_MUTEX_INITIALIZER;

#ifdef _DEBUG_
void display_session( struct session *s )
{
   if (!s) return;

    fprintf( stdout, "  [DEBUG]  Session %ld chat \n", s->id );
    for (int i = 0; i < s->message_count; i++) {
        if (s->messages[i].content)
            fprintf( stdout, "  (%d)  Role: \"%s\"  message: \"%s\"\n", i,
                      s->messages[i].role == ROLE_SYSTEM ? "SYSTEM" :
                      s->messages[i].role == ROLE_USER ? "USER" :
                      s->messages[i].role == ROLE_ASSISTANT ? "ASSISTANT" :
                      "(unknown)", 
                      s->messages[i].content );
    }
}
#endif

static uint64_t now_sec(void)
{
    return (uint64_t)time(NULL);
}

/* internal cleanup */

static void session_cleanup(struct session *s)
{
    for (int i = 0; i < s->message_count; i++) {
        if (s->messages[i].content)
            free(s->messages[i].content);
    }

    s->message_count = 0;

    if (s->kv_k) free(s->kv_k);
    if (s->kv_v) free(s->kv_v);

    s->kv_k = NULL;
    s->kv_v = NULL;

    s->token_count = 0;
    s->kv_len = 0;
}

/* initialize manager */

void session_manager_init(void)
{
    pthread_mutex_lock(&table_lock);

    for (int i = 0; i < NS; i++) {

        memset(&sessions[i], 0, sizeof(struct session));

        pthread_mutex_init(&sessions[i].lock, NULL);
        sessions[i].valid = 0;
    }

    pthread_mutex_unlock(&table_lock);
}

/* locate oldest session (inactive) */

static int find_oldest_session(void)
{
    uint64_t oldest = 0;
    int index = -1;

    for (int i = 0; i < NS; i++) {

        if (!sessions[i].valid)
            return i;

        if (pthread_mutex_trylock(&sessions[i].lock) != 0)
            continue;

        if (index < 0 || sessions[i].last_access < oldest) {
            oldest = sessions[i].last_access;
            index = i;
        }

        pthread_mutex_unlock(&sessions[i].lock);
    }

    return index;
}

/* create new session */

uint64_t session_create(void)
{
    pthread_mutex_lock(&table_lock);

    int slot = find_oldest_session();

    if (slot < 0) {
       fprintf( stdout, " [Error]  Could not obtain a session handle\n" );
        pthread_mutex_unlock(&table_lock);
        return 0;
    }
    fprintf( stdout, " [DEBUG]  Obtained session handle (slot): %d\n", slot );

    struct session *s = &sessions[slot];

    if (s->valid) {

        pthread_mutex_lock(&s->lock);

        session_cleanup(s);

        pthread_mutex_unlock(&s->lock);
    }

    uint64_t id = ((uint64_t)rand() << 32) ^ rand();

    s->id = id;
    s->valid = 1;
    fprintf( stdout, " [DEBUG]  Session ID: %ld\n", id );

    s->created_at = now_sec();
    s->last_access = s->created_at;

    s->token_count = 0;
    s->kv_len = 0;
    s->message_count = 0;

    s->kv_k = NULL;
    s->kv_v = NULL;

    pthread_mutex_unlock(&table_lock);

    return id;
}

/* lookup session and attach by locking the mutex */

struct session *session_attach(uint64_t id)
{
    pthread_mutex_lock(&table_lock);

    for (int i = 0; i < NS; i++) {

        if (!sessions[i].valid)
            continue;

        if (sessions[i].id == id) {

            struct session *s = &sessions[i];

            pthread_mutex_lock(&s->lock);

            s->last_access = now_sec();

            pthread_mutex_unlock(&table_lock);

            return s;
        }
    }

    pthread_mutex_unlock(&table_lock);

    return NULL;
}

/* delete session */

int session_delete(uint64_t id)
{
    pthread_mutex_lock(&table_lock);

    fprintf( stdout, " [DEBUG]  About to delete session with ID: %ld\n", id );
    for (int i = 0; i < NS; i++) {
       fprintf( stdout, "  [%d]  session id: %ld, valid: %d\n", i,
                sessions[i].id, sessions[i].valid );

        if (!sessions[i].valid)
            continue;

        if (sessions[i].id == id) {

            struct session *s = &sessions[i];

            pthread_mutex_lock(&s->lock);

            session_cleanup(s);

            s->valid = 0;

            pthread_mutex_unlock(&s->lock);
            pthread_mutex_unlock(&table_lock);

            fprintf( stdout, " [DEBUG]  Session %d, id: %ld deleted\n", i, id );
            return 0;
        }
    }

    pthread_mutex_unlock(&table_lock);

    fprintf( stdout, " [Error]  Failed to delete session with id: %ld \n", id );
    return -1;
}

/* update access timestamp */

void session_touch(struct session *s)
{
    if (!s)
        return;

    s->last_access = now_sec();
}

