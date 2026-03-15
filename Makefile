CC = gcc
COPTS = -Wall -g
 COPTS += -D _DEBUG_

all:
	$(CC) $(COPTS) \
        session_manager.c \
        json_parser.c \
        http_server.c \
        chat.c \
        prompt_builder.c \
            main.c -lpthread

