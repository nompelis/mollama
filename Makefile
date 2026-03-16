CC = gcc
COPTS = -Wall -g
 COPTS += -D _DEBUG_
 COPTS += -D _DEBUG2_

all:
	$(CC) $(COPTS) \
        session_manager.c \
        json_parser.c \
        http_server.c \
        chat.c \
        prompt_builder.c \
        tokenizer.c \
        inference.c \
        vocab.c \
        bpe.c \
            main.c -lpthread

