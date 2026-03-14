#ifndef JSON_H
#define JSON_H

#include <stdint.h>

struct json_request {

    uint64_t session_id;

    char prompt[4096];

    int has_session;
};

/* parsing */

int json_parse_chat(
    const char *body,
    struct json_request *req
);

#endif

