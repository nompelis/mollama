CC = gcc
COPTS = -Wall -g

all:
	$(CC) session_manager.c json_parser.c http_server.c main.c -lpthread

