#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <pthread.h>

#include "config.h"
#include "json.h"
#include "session.h"
#include "chat.h"
#include "prompt_builder.h"
#include "inference.h"

static struct inference_engine *g_engine = NULL;

struct worker_arg {
    int client_fd;
};

static int send_all(int fd, const char *buf, size_t len)
{
    size_t off = 0;

    while (off < len) {
        ssize_t n = send(fd, buf + off, len - off, 0);

        if (n < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }

        if (n == 0)
            return -1;

        off += (size_t)n;
    }

    return 0;
}

static int send_text_response(
    int fd,
    int status_code,
    const char *status_text,
    const char *content_type,
    const char *body)
{
    char header[1024];
    size_t body_len = strlen(body);

    int n = snprintf(
        header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        status_code, status_text, content_type, body_len
    );

    if (n < 0 || (size_t)n >= sizeof(header))
        return -1;

    if (send_all(fd, header, (size_t)n) != 0)
        return -1;

    if (send_all(fd, body, body_len) != 0)
        return -1;

    return 0;
}

static int send_json_response(int fd, int status_code, const char *status_text, const char *json)
{
    return send_text_response(fd, status_code, status_text, "application/json", json);
}

static int send_error_json(int fd, int status_code, const char *status_text, const char *msg)
{
    char body[512];

    snprintf(body, sizeof(body),
        "{"
        "\"error\":\"%s\""
        "}",
        msg);

    return send_json_response(fd, status_code, status_text, body);
}

static int send_stream_header(int fd)
{
    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/x-ndjson\r\n"
        "Connection: close\r\n"
        "\r\n";

    return send_all(fd, hdr, strlen(hdr));
}

static int send_stream_fragment(int fd, const char *model, const char *content, int done)
{
    char line[1024];

    int n = snprintf(
        line, sizeof(line),
        "{"
        "\"model\":\"%s\","
        "\"message\":{"
        "\"role\":\"assistant\","
        "\"content\":\"%s\""
        "},"
        "\"done\":%s"
        "}\n",
        model,
        content,
        done ? "true" : "false"
    );

    if (n < 0 || (size_t)n >= sizeof(line))
        return -1;

    return send_all(fd, line, (size_t)n);
}

static const char *find_header_value(const char *headers, const char *key)
{
    size_t key_len = strlen(key);
    const char *p = headers;

    while ((p = strstr(p, key)) != NULL) {
        if ((p == headers || p[-1] == '\n') && strncmp(p, key, key_len) == 0)
            return p + key_len;
        p += key_len;
    }

    return NULL;
}

static int parse_content_length(const char *headers, size_t *out_len)
{
    const char *p = find_header_value(headers, "Content-Length:");

    if (!p)
        return -1;

    while (*p == ' ' || *p == '\t')
        p++;

    char *end = NULL;
    unsigned long v = strtoul(p, &end, 10);

    if (end == p)
        return -1;

    *out_len = (size_t)v;
    return 0;
}

static int read_http_request(int fd, char *buf, size_t cap, size_t *out_len)
{
    size_t used = 0;

    for (;;) {
        if (used + 1 >= cap)
            return -1;

        ssize_t n = recv(fd, buf + used, cap - used - 1, 0);

        if (n < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }

        if (n == 0)
            break;

        used += (size_t)n;
        buf[used] = '\0';

        char *hdr_end = strstr(buf, "\r\n\r\n");
        if (!hdr_end)
            continue;

        size_t header_len = (size_t)(hdr_end - buf) + 4;
        size_t content_len = 0;

        if (parse_content_length(buf, &content_len) != 0)
            content_len = 0;

        if (header_len + content_len <= used) {
            *out_len = used;
            return 0;
        }
    }

    return -1;
}

static int extract_method_path(const char *req, char *method, size_t msz, char *path, size_t psz)
{
    const char *sp1 = strchr(req, ' ');
    if (!sp1)
        return -1;

    const char *sp2 = strchr(sp1 + 1, ' ');
    if (!sp2)
        return -1;

    size_t mlen = (size_t)(sp1 - req);
    size_t plen = (size_t)(sp2 - (sp1 + 1));

    if (mlen == 0 || mlen >= msz || plen == 0 || plen >= psz)
        return -1;

    memcpy(method, req, mlen);
    method[mlen] = '\0';

    memcpy(path, sp1 + 1, plen);
    path[plen] = '\0';

    return 0;
}

static char *get_body_ptr(char *req)
{
    char *p = strstr(req, "\r\n\r\n");
    if (!p)
        return NULL;
    return p + 4;
}

static int parse_session_id_from_path(const char *path, uint64_t *out_id)
{
    const char *prefix = "/api/session/";
    size_t prefix_len = strlen(prefix);

    if (strncmp(path, prefix, prefix_len) != 0)
        return -1;

    const char *p = path + prefix_len;
    if (*p == '\0')
        return -1;

    char *end = NULL;
    unsigned long long v = strtoull(p, &end, 10);

    if (end == p || *end != '\0')
        return -1;

    *out_id = (uint64_t)v;
    return 0;
}

static int handle_post_session(int fd)
{
    uint64_t id = session_create();

    if (id == 0)
        return send_error_json(fd, 503, "Service Unavailable", "session_unavailable");

    char body[256];
    snprintf(body, sizeof(body),
        "{"
        "\"session_id\":%llu"
        "}",
        (unsigned long long)id);

    return send_json_response(fd, 200, "OK", body);
}

static int handle_delete_session(int fd, const char *path)
{
    uint64_t id = 0;

    if (parse_session_id_from_path(path, &id) != 0)
        return send_error_json(fd, 400, "Bad Request", "bad_session_path");

    if (session_delete(id) != 0)
        return send_error_json(fd, 404, "Not Found", "invalid_session");

    return send_json_response(fd, 200, "OK", "{\"deleted\":true}");
}


struct stream_ctx {
    int fd;
    char *buffer;
    size_t len;
    size_t cap;
};

static int stream_buf_append(struct stream_ctx *ctx, const char *text)
{
    size_t n = strlen(text);

    if (ctx->len + n + 1 > ctx->cap) {

        size_t newcap = ctx->cap ? ctx->cap * 2 : 1024;

        while (newcap < ctx->len + n + 1)
            newcap *= 2;

        char *p = realloc(ctx->buffer, newcap);
        if (!p)
            return -1;

        ctx->buffer = p;
        ctx->cap = newcap;
    }

    memcpy(ctx->buffer + ctx->len, text, n);

    ctx->len += n;
    ctx->buffer[ctx->len] = '\0';

    return 0;
}

static int stream_callback(
    const char *text,
    int done,
    void *user
)
{
    struct stream_ctx *ctx = user;

    if (stream_buf_append(ctx, text) != 0)
        return -1;

    /* stream to client */

    if (send_stream_fragment(ctx->fd, "local", text, done) != 0)
        return -1;

    return 0;
}

static int handle_post_chat(int fd, char *body)
{
    fprintf( stdout, " [DEBUG]  Got HTTP body: -->%s<--\n", body );
    struct json_request req;

    if (json_parse_chat(body, &req) != 0)
        return send_error_json(fd, 400, "Bad Request", "bad_json");

    if (!req.has_session)
        return send_error_json(fd, 400, "Bad Request", "missing_session_id");

    struct session *s = session_attach(req.session_id);
    if (!s)
        return send_error_json(fd, 404, "Not Found", "invalid_session");

    /* store user message */

    if (chat_add_user_message(s, req.prompt) != 0) {
        pthread_mutex_unlock(&s->lock);
        return send_error_json(fd, 409, "Conflict", "message_limit_exceeded");
    }

    char prompt[16384];
    if (prompt_build(s, prompt, sizeof(prompt)) < 0) {
        pthread_mutex_unlock(&s->lock);
        return send_error_json(fd, 500, "Internal Server Error", "prompt_build_failed");
    }
#ifdef _DEBUG_
    display_session( s );
    fprintf( stdout, " [DEBUG]  Prompt:\n-----\n%s\n-----\n", prompt );
#endif

    if (inference_update_prompt_tokens(g_engine, s, prompt) < 0) {
        pthread_mutex_unlock(&s->lock);
        return send_error_json(fd, 500, "Internal Server Error", "token_update_failed");
    }

    if (s->prompt_token_count > MAX_CONTEXT) {
        pthread_mutex_unlock(&s->lock);
        return send_error_json(
            fd,
            409,
            "Conflict",
            "context_length_exceeded"
        );
    }

    /* stream response as it is generated fragment-by-fragment */

    s->generated_token_count = 0;

    struct stream_ctx ctx = { .fd=fd, .len=0, .cap=0, .buffer=NULL };

    if (send_stream_header(fd) != 0) {
        pthread_mutex_unlock(&s->lock);
        return -1;
    }

#ifdef _DEBUG_
    fprintf( stdout, " [DEBUG]  Http server calling inference/streaming\n" );
#endif
    inference_generate(g_engine, s, prompt, stream_callback, &ctx);

    /* store assistant message */

    chat_add_assistant_message(s, ctx.buffer ? ctx.buffer : "");

    if (ctx.buffer) free(ctx.buffer);

    pthread_mutex_unlock(&s->lock);

    return 0;
}

static void *worker_main(void *arg)
{
    struct worker_arg *wa = (struct worker_arg *)arg;
    int fd = wa->client_fd;

    free(wa);

    char reqbuf[MAX_JSON_BODY + 4096];
    size_t req_len = 0;

    if (read_http_request(fd, reqbuf, sizeof(reqbuf), &req_len) != 0) {
        send_error_json(fd, 400, "Bad Request", "bad_request");
        close(fd);
        return NULL;
    }

    char method[16];
    char path[256];

    if (extract_method_path(reqbuf, method, sizeof(method), path, sizeof(path)) != 0) {
        send_error_json(fd, 400, "Bad Request", "bad_request_line");
        close(fd);
        return NULL;
    }

    if (strcmp(method, "POST") == 0 && strcmp(path, "/api/session") == 0) {
        handle_post_session(fd);
        close(fd);
        return NULL;
    }

    if (strcmp(method, "DELETE") == 0 && strncmp(path, "/api/session/", 13) == 0) {
        handle_delete_session(fd, path);
        close(fd);
        return NULL;
    }

    if (strcmp(method, "POST") == 0 && strcmp(path, "/api/chat") == 0) {
        char *body = get_body_ptr(reqbuf);

        if (!body) {
            send_error_json(fd, 400, "Bad Request", "missing_body");
            close(fd);
            return NULL;
        }

        handle_post_chat(fd, body);
        close(fd);
        return NULL;
    }

    send_error_json(fd, 404, "Not Found", "unknown_endpoint");
    close(fd);
    return NULL;
}

int http_server_run(struct inference_engine *engine)
{
    g_engine = engine;

    int server_fd = -1;
    int opt = 1;
    struct sockaddr_in addr;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0)
        return -1;

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
        close(server_fd);
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(SERVER_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 16) != 0) {
        close(server_fd);
        return -1;
    }

    for (;;) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) {
            if (errno == EINTR)
                continue;
            break;
        }

        struct worker_arg *wa = malloc(sizeof(*wa));
        if (!wa) {
            close(client_fd);
            continue;
        }

        wa->client_fd = client_fd;

        pthread_t tid;
        if (pthread_create(&tid, NULL, worker_main, wa) != 0) {
            free(wa);
            close(client_fd);
            continue;
        }

        pthread_detach(tid);
    }

    close(server_fd);
    return -1;
}

