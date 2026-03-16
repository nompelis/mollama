#include <stdio.h>
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
  token_id tokens[100];//HACK
  tokenizer_encode(tok, "This is a shim.", tokens, 4096);//HACK

    struct inference_engine *engine = inference_create(tok);
    if (!engine)
        return 1;

    fprintf( stdout, " [DEBUG]  Driver starting the server \n" );
    return http_server_run(engine);
}

