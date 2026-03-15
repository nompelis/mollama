#ifndef CHAT_H
#define CHAT_H

#include "session.h"
#include "roles.h"

int chat_add_user_message(
    struct session *s,
    const char *text
);

int chat_add_assistant_message(
    struct session *s,
    const char *text
);

#endif

