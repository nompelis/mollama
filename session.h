#ifndef SESSION_H
#define SESSION_H

#include <stdint.h>
#include <pthread.h>
#include "config.h"
#include "roles.h"

struct session {
    uint64_t id;
    int valid;

    pthread_mutex_t lock;

    uint64_t created_at;
    uint64_t last_access;

    /* token context */

    int tokens[MAX_CONTEXT];
    int token_count;

    /* KV cache */

    float *kv_k;
    float *kv_v;
    int kv_len;

    /* chat history */

    struct message messages[MAX_MESSAGES];
    int message_count;
    int prompt_token_count;
};

/* lifecycle */

void session_manager_init(void);

uint64_t session_create(void);

struct session *session_attach(uint64_t id);

int session_delete(uint64_t id);

/* utilities */

void session_touch(struct session *s);

#ifdef _DEBUG_
void display_session( struct session *s );
#endif

#endif

