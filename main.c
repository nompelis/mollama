#include <stdio.h>
#include "tokenizer.h"
#include "transformer.h"
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

    struct model_config cfg = {
       .vocab_size = tokenizer_vocab_size(tok),
       .context_length = 2048,
       .n_layers = 1,
       .n_heads = 1,
       .hidden_size = 10,
       .head_size = 10,
       .ff_size = 20
    };
    struct transformer *trans = transformer_create(&cfg);
    if (!trans)
        return 1;

    struct inference_engine *engine = inference_create(tok, trans);
    if (!engine)
        return 1;

    fprintf( stdout, " [DEBUG]  Driver starting the server \n" );
    return http_server_run(engine);
}

