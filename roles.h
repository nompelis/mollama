#ifndef ROLES_H
#define ROLES_H

#include <stdint.h>

enum role {
    ROLE_SYSTEM = 0,
    ROLE_USER = 1,
    ROLE_ASSISTANT = 2
};

struct message {
    uint8_t role;
    char *content;
    int token_count;
};

#endif

