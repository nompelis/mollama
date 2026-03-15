#include "tokenizer.h"
#include "inference.h"
#include "session.h"
#include "http_server.h"

int main(void)
{
    session_manager_init();

    struct tokenizer *tok = tokenizer_create(NULL);
    if (!tok)
        return 1;

    struct inference_engine *engine = inference_create(tok);
    if (!engine)
        return 1;

    return http_server_run(engine);
}

