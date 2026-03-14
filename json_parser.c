#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "json.h"
#include "config.h"

/* skip whitespace */

static const char *skip_ws(const char *p)
{
    while (*p && isspace((unsigned char)*p))
        p++;

    return p;
}

/* find JSON key */

static const char *find_key(const char *json, const char *key)
{
    const char *p = json;
    size_t klen = strlen(key);

    while ((p = strstr(p, key))) {

        const char *q = p - 1;

        if (q >= json && *q == '"')
            return p;

        p += klen;
    }

    return NULL;
}

/* parse uint64 */

static int parse_uint64(const char *p, uint64_t *value)
{
    p = skip_ws(p);

    if (!isdigit((unsigned char)*p))
        return -1;

    char *end;

    *value = strtoull(p, &end, 10);

    return 0;
}

/* parse JSON "message" string */

static int parse_message(const char *p, char *out, size_t maxlen)
{
    p = skip_ws(p);

    if (*p != '"')
        return -1;

    p++;

    size_t i = 0;

    while (*p) {

        if (*p == '\\') {

            p++;

            if (*p == '"' || *p == '\\') {

                if (i + 1 >= maxlen)
                    return -1;

                out[i++] = *p++;
                continue;
            }

            return -1;
        }

        if (*p == '"')
            break;

        if (i + 1 >= maxlen)
            return -1;

        out[i++] = *p++;
    }

    if (*p != '"')
        return -1;

    out[i] = '\0';

    return 0;
}


int json_parse_chat(const char *body, struct json_request *req)
{
    memset(req, 0, sizeof(*req));

    const char *p;

    /* session_id */

    p = find_key(body, "session_id");

    if (p) {

        p = strchr(p, ':');

        if (!p)
            return -1;

        p++;

        if (parse_uint64(p, &req->session_id) != 0)
            return -1;

        req->has_session = 1;
    }

    /* message */

    p = find_key(body, "message");

    if (!p)
        return -1;

    p = strchr(p, ':');

    if (!p)
        return -1;

    p++;

    if (parse_message(p, req->prompt, MAX_PROMPT) != 0)
        return -1;

    return 0;
}
