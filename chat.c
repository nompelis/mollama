#include <stdlib.h>
#include <string.h>

#include "session.h"
#include "roles.h"

static int chat_add_message(struct session *s, uint8_t role, const char *text)
{
    struct message *m;
    size_t len;
    char *copy;

    if (!s || !text)
        return -1;

    if (s->message_count >= MAX_MESSAGES)
        return -1;

    len = strlen(text);

    copy = (char *)malloc(len + 1);
    if (!copy)
        return -1;

    memcpy(copy, text, len + 1);

    m = &s->messages[s->message_count];

    m->role = role;
    m->content = copy;
    m->token_count = 0;

    s->message_count++;

    return 0;
}

int chat_add_user_message(struct session *s, const char *text)
{
    return chat_add_message(s, ROLE_USER, text);
}

int chat_add_assistant_message(struct session *s, const char *text)
{
    return chat_add_message(s, ROLE_ASSISTANT, text);
}

