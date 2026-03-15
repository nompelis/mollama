#ifndef PROMPT_BUILDER_H
#define PROMPT_BUILDER_H

#include <stddef.h>
#include "session.h"

int prompt_build(
    struct session *s,
    char *out,
    size_t out_size
);

#endif

