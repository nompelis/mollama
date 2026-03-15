#include <stdio.h>
#include <string.h>

#include "prompt_builder.h"
#include "roles.h"

static const char *role_tag(uint8_t role)
{
    switch (role) {

        case ROLE_SYSTEM:
            return "system";

        case ROLE_USER:
            return "user";

        case ROLE_ASSISTANT:
            return "assistant";

        default:
            return "unknown";
    }
}

int prompt_build(
    struct session *s,
    char *out,
    size_t out_size
)
{
    size_t pos = 0;

    if (!s || !out || out_size == 0)
        return -1;

    out[0] = '\0';

    for (int i = 0; i < s->message_count; i++) {

        struct message *m = &s->messages[i];

        const char *tag = role_tag(m->role);

        int n = snprintf(
            out + pos,
            out_size - pos,
            "<%s>\n%s\n</%s>\n\n",
            tag,
            m->content,
            tag
        );

        if (n < 0 || (size_t)n >= out_size - pos)
            return -1;

        pos += (size_t)n;
    }

    /* open assistant block for generation */

    int n = snprintf(
        out + pos,
        out_size - pos,
        "<assistant>\n"
    );

    if (n < 0 || (size_t)n >= out_size - pos)
        return -1;

    pos += (size_t)n;

    return (int)pos;
}

