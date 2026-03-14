#include "session.h"
#include "http_server.h"

int main(void)
{
    session_manager_init();
    return http_server_run();
}

