/*
 * Unit tests for the FMU C wrapper.  Includes the wrapper source so the
 * static helpers (framing, JSON number parsing, endpoint discovery) are
 * testable directly; the FMI entry points are exercised against a fake
 * sidecar on a socketpair and, for instantiation, a real loopback
 * listener on a thread.  Built and run by tests/fmi/test_c_unit.py,
 * plain and under -fsanitize=address,undefined.
 */

#include "../../../src/maddening/fmi/c/maddening_fmu.c"

#include <assert.h>
#include <pthread.h>
#include <stdint.h>

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failures; \
    fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

/* ------------------------------------------------------------ helpers */

static char g_last_log[4096];
static int g_log_calls = 0;
static void test_logger(fmi3InstanceEnvironment env, fmi3Status st, fmi3String cat, fmi3String msg) {
    (void)env; (void)st; (void)cat;
    ++g_log_calls;
    snprintf(g_last_log, sizeof g_last_log, "%s", msg ? msg : "");
}

static Instance *fake_instance(sock_t s) {
    Instance *in = (Instance *)calloc(1, sizeof *in);
    in->sock = s; in->log = test_logger;
    return in;
}

static void free_instance(Instance *in) {
    free(in->req); free(in->resp); free(in);
}

/* Fake sidecar: reads one framed request from `s`, returns it (malloc'd),
 * and writes `reply` (framed) back.  reply == NULL closes the socket. */
static char *fake_exchange(sock_t s, const char *reply, size_t reply_len_override) {
    unsigned char head[4];
    if (recv_all(s, (char *)head, 4)) return NULL;
    size_t n = ((size_t)head[0] << 24) | ((size_t)head[1] << 16) | ((size_t)head[2] << 8) | head[3];
    char *req = (char *)malloc(n + 1);
    if (recv_all(s, req, n)) { free(req); return NULL; }
    req[n] = '\0';
    if (reply == NULL) { sock_close(s); return req; }
    size_t m = reply_len_override ? reply_len_override : strlen(reply);
    unsigned char h2[4] = { (unsigned char)(m >> 24), (unsigned char)(m >> 16),
                            (unsigned char)(m >> 8), (unsigned char)m };
    send_all(s, (const char *)h2, 4);
    send_all(s, reply, strlen(reply));   /* may be shorter than advertised */
    if (reply_len_override) sock_close(s);   /* truncated frame: hang up */
    return req;
}

typedef struct { sock_t s; const char *reply; size_t len_override; char *seen; } ServerArgs;
static void *server_thread(void *p) {
    ServerArgs *a = (ServerArgs *)p;
    a->seen = fake_exchange(a->s, a->reply, a->len_override);
    return NULL;
}

/* Run `body` on the client end while the fake server answers once; the
 * request the server saw is left in g_seen for the caller to inspect. */
static char *g_seen = NULL;
#define WITH_SERVER(reply, len_override, body) do {                              \
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);              \
    ServerArgs args = { sv[1], (reply), (len_override), NULL };                  \
    pthread_t th; pthread_create(&th, NULL, server_thread, &args);               \
    Instance *in = fake_instance(sv[0]);                                         \
    body;                                                                        \
    pthread_join(th, NULL);                                                      \
    sock_close(sv[0]); if ((reply) != NULL && !(len_override)) sock_close(sv[1]); \
    free(g_seen); g_seen = args.seen ? args.seen : strdup("");                   \
    free_instance(in);                                                           \
} while (0)

/* ------------------------------------------------------- parse_values */

static void test_parse_values(void) {
    Instance *in = fake_instance(SOCK_INVALID);
    double out[4];
    in->resp = strdup("{\"ok\":true,\"values\":[1.5, -2e3 ,3,4.25]}");
    CHECK(parse_values(in, out, 4) == fmi3OK);
    CHECK(out[0] == 1.5 && out[1] == -2000.0 && out[2] == 3.0 && out[3] == 4.25);
    CHECK(parse_values(in, out, 0) == fmi3OK);
    /* too few values */
    CHECK(parse_values(in, out, 5) == fmi3Error);
    free(in->resp);
    in->resp = strdup("{\"ok\":true}");
    CHECK(parse_values(in, out, 1) == fmi3Error);
    free(in->resp);
    in->resp = strdup("{\"ok\":true,\"values\":[abc]}");
    CHECK(parse_values(in, out, 1) == fmi3Error);
    free(in->resp);
    in->resp = strdup("{\"ok\":true,\"values\":[]}");
    CHECK(parse_values(in, out, 1) == fmi3Error);
    CHECK(parse_values(in, out, 0) == fmi3OK);
    free(in->resp);
    /* %.17g round trip of awkward doubles */
    in->resp = strdup("{\"ok\":true,\"values\":[0.1,1e-300,1.7976931348623157e308,-0.0]}");
    CHECK(parse_values(in, out, 4) == fmi3OK);
    CHECK(out[0] == 0.1 && out[1] == 1e-300 && out[2] == 1.7976931348623157e308);
    free(in->resp); in->resp = NULL;
    free_instance(in);
}

/* ------------------------------------------------------ read_endpoint */

static void test_read_endpoint(void) {
    char host[256]; int port = 0;
    unsetenv("MADDENING_FMU_ENDPOINT");
    CHECK(read_endpoint(NULL, host, sizeof host, &port) == -1);
    CHECK(read_endpoint("/nonexistent/dir", host, sizeof host, &port) == -1);
    CHECK(read_endpoint("file://", host, sizeof host, &port) == -1);   /* empty after prefix */
    CHECK(read_endpoint("", host, sizeof host, &port) == -1);
    setenv("MADDENING_FMU_ENDPOINT", "localhost:5555", 1);
    CHECK(read_endpoint(NULL, host, sizeof host, &port) == 0);
    CHECK(strcmp(host, "localhost") == 0 && port == 5555);
    setenv("MADDENING_FMU_ENDPOINT", "nocolon", 1);
    CHECK(read_endpoint(NULL, host, sizeof host, &port) == -1);
    setenv("MADDENING_FMU_ENDPOINT", "h:0", 1);
    CHECK(read_endpoint(NULL, host, sizeof host, &port) == -1);
    setenv("MADDENING_FMU_ENDPOINT", "::1:6000", 1);      /* last colon wins */
    CHECK(read_endpoint(NULL, host, sizeof host, &port) == 0 && strcmp(host, "::1") == 0 && port == 6000);
    setenv("MADDENING_FMU_ENDPOINT", "[::1]:6001", 1);    /* bracketed IPv6 literal */
    CHECK(read_endpoint(NULL, host, sizeof host, &port) == 0 && strcmp(host, "::1") == 0 && port == 6001);
    setenv("MADDENING_FMU_ENDPOINT", "[]:6001", 1);
    CHECK(read_endpoint(NULL, host, sizeof host, &port) == -1);
    /* host longer than the buffer is refused, not truncated */
    char big[600]; memset(big, 'a', 590); memcpy(big + 590, ":1234", 6);
    setenv("MADDENING_FMU_ENDPOINT", big, 1);
    CHECK(read_endpoint(NULL, host, sizeof host, &port) == -1);
    unsetenv("MADDENING_FMU_ENDPOINT");
    /* resource file, with and without trailing slash and file:// */
    char dir[] = "/tmp/maddening_fmu_test_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char path[512]; snprintf(path, sizeof path, "%s/endpoint.txt", dir);
    FILE *f = fopen(path, "w"); fputs("127.0.0.1:4321\n", f); fclose(f);
    CHECK(read_endpoint(dir, host, sizeof host, &port) == 0 && port == 4321);
    char withslash[512]; snprintf(withslash, sizeof withslash, "%s/", dir);
    CHECK(read_endpoint(withslash, host, sizeof host, &port) == 0 && port == 4321);
    char uri[512]; snprintf(uri, sizeof uri, "file://%s", dir);
    CHECK(read_endpoint(uri, host, sizeof host, &port) == 0 && strcmp(host, "127.0.0.1") == 0);
    remove(path); rmdir(dir);
}

/* -------------------------------------------------------- bridge_call */

static void test_bridge_call_paths(void) {
    WITH_SERVER("{\"ok\":true,\"t\":1}", 0, {
        CHECK(bridge_call(in, "{\"op\":\"hello\"}") == fmi3OK);
        CHECK(strcmp(in->resp, "{\"ok\":true,\"t\":1}") == 0);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"hello\"}") == 0);
    g_log_calls = 0;
    WITH_SERVER("{\"ok\":false,\"error\":\"KeyError: boom\"}", 0, {
        CHECK(bridge_call(in, "{\"op\":\"x\"}") == fmi3Error);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "KeyError: boom") != NULL);
    /* server closes without replying */
    WITH_SERVER(NULL, 0, {
        CHECK(bridge_call(in, "{\"op\":\"x\"}") == fmi3Error);
    });
    /* header advertises more bytes than sent: recv fails, no read past buffer */
    WITH_SERVER("{\"ok\":true}", 4096, {
        CHECK(bridge_call(in, "{\"op\":\"x\"}") == fmi3Error);
    });
    /* zero-length body */
    WITH_SERVER("", 0, {
        CHECK(bridge_call(in, "{\"op\":\"x\"}") == fmi3Error);
        CHECK(in->resp != NULL && in->resp[0] == '\0');
    });
    /* a reply larger than the initial buffer grows it */
    char *big = (char *)malloc(70000);
    memcpy(big, "{\"ok\":true,\"pad\":\"", 18);
    memset(big + 18, 'x', 69000); memcpy(big + 69018, "\"}", 3);
    WITH_SERVER(big, 0, {
        CHECK(bridge_call(in, "{\"op\":\"x\"}") == fmi3OK);
        CHECK(in->resp_cap >= 69021 && strlen(in->resp) == 69020);
    });
    free(big);
    /* an unusable socket */
    Instance *dead = fake_instance(SOCK_INVALID);
    CHECK(bridge_call(dead, "{}") == fmi3Error);
    free_instance(dead);
    /* the peer is gone before we send: EPIPE -> fmi3Error, not SIGPIPE */
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    sock_close(sv[1]);
    Instance *gone = fake_instance(sv[0]);
    g_log_calls = 0;
    CHECK(bridge_call(gone, "{\"op\":\"hello\"}") == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "send failed") != NULL);
    sock_close(sv[0]); free_instance(gone);
}

/* --------------------------------------------------- get / set / step */

static void test_get_set_step(void) {
    fmi3ValueReference vr[3] = { 7, 9, 11 };
    fmi3Float64 v64[3] = { 1.5, -2.0, 1e-9 };
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetFloat64((fmi3Instance)in, vr, 3, v64, 3) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"set\",\"vr\":[7,9,11],\"values\":[1.5,-2,1.0000000000000001e-09]}") == 0);
    fmi3Int32 vi[2] = { -5, 7 };
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetInt32((fmi3Instance)in, vr, 2, vi, 2) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"set\",\"vr\":[7,9],\"values\":[-5,7]}") == 0);
    fmi3Boolean vb[1] = { fmi3True };
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetBoolean((fmi3Instance)in, vr, 1, vb, 1) == fmi3OK);
    });
    /* NaN / inf never leave the process */
    fmi3Float32 bad[1] = { NAN };
    Instance *dead = fake_instance(SOCK_INVALID);
    g_log_calls = 0;
    CHECK(fmi3SetFloat32((fmi3Instance)dead, vr, 1, bad, 1) == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "non-finite") != NULL);
    /* nValues == 0 needs no socket */
    CHECK(fmi3SetFloat64((fmi3Instance)dead, vr, 0, v64, 0) == fmi3OK);
    CHECK(fmi3GetFloat64((fmi3Instance)dead, vr, 0, v64, 0) == fmi3OK);
    CHECK(fmi3SetFloat64(NULL, vr, 1, v64, 1) == fmi3Error);
    free_instance(dead);
    /* getters convert every width */
    fmi3Float32 g32[2]; fmi3UInt8 g8[2]; fmi3Int64 g64[2]; fmi3Boolean gb[2];
    WITH_SERVER("{\"ok\":true,\"values\":[2.5,-1]}", 0, {
        CHECK(fmi3GetFloat32((fmi3Instance)in, vr, 2, g32, 2) == fmi3OK);
        CHECK(g32[0] == 2.5f && g32[1] == -1.0f);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"get\",\"vr\":[7,9]}") == 0);
    WITH_SERVER("{\"ok\":true,\"values\":[200,255]}", 0, {
        CHECK(fmi3GetUInt8((fmi3Instance)in, vr, 2, g8, 2) == fmi3OK && g8[0] == 200 && g8[1] == 255);
    });
    WITH_SERVER("{\"ok\":true,\"values\":[-9007199254740992,1]}", 0, {
        CHECK(fmi3GetInt64((fmi3Instance)in, vr, 2, g64, 2) == fmi3OK && g64[0] == -9007199254740992LL);
    });
    WITH_SERVER("{\"ok\":true,\"values\":[1,0]}", 0, {
        CHECK(fmi3GetBoolean((fmi3Instance)in, vr, 2, gb, 2) == fmi3OK && gb[0] == fmi3True && gb[1] == fmi3False);
    });
    /* reply with fewer values than requested: error, outputs untouched */
    g32[0] = 42.0f;
    WITH_SERVER("{\"ok\":true,\"values\":[1]}", 0, {
        CHECK(fmi3GetFloat32((fmi3Instance)in, vr, 2, g32, 2) == fmi3Error);
    });
    /* DoStep bookkeeping */
    fmi3Boolean ev, term, early; fmi3Float64 last;
    WITH_SERVER("{\"ok\":true,\"t\":0.25}", 0, {
        CHECK(fmi3DoStep((fmi3Instance)in, 0.2, 0.05, fmi3False, &ev, &term, &early, &last) == fmi3OK);
        CHECK(last == 0.25 && in->time == 0.25 && !ev && !term && !early);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"step\",\"t\":0.20000000000000001,\"dt\":0.05000000000000000277}") == 0
          || strstr(g_seen, "\"op\":\"step\"") != NULL);
    WITH_SERVER("{\"ok\":false,\"error\":\"x\"}", 0, {
        in->time = 0.2;
        CHECK(fmi3DoStep((fmi3Instance)in, 0.2, 0.05, fmi3False, &ev, &term, &early, &last) == fmi3Error);
        CHECK(last == 0.2);
    });
}

/* ----------------------------------------------------------- FMU state */

static void test_fmu_state(void) {
    fmi3FMUState st = NULL;
    WITH_SERVER("{\"ok\":true,\"state\":\"QUJDRA==\"}", 0, {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK);
    });
    CHECK(st != NULL && ((FmuState *)st)->n == 8 && memcmp(((FmuState *)st)->blob, "QUJDRA==", 8) == 0);
    size_t n = 0;
    CHECK(fmi3SerializedFMUStateSize(NULL, st, &n) == fmi3OK && n == 8);
    fmi3Byte buf[8];
    CHECK(fmi3SerializeFMUState(NULL, st, buf, 4) == fmi3Error);      /* too small */
    CHECK(fmi3SerializeFMUState(NULL, st, buf, 8) == fmi3OK);
    fmi3FMUState st2 = NULL;
    CHECK(fmi3DeserializeFMUState(NULL, buf, 8, &st2) == fmi3OK);
    CHECK(memcmp(((FmuState *)st2)->blob, "QUJDRA==", 8) == 0);
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetFMUState((fmi3Instance)in, st2) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"set_state\",\"state\":\"QUJDRA==\"}") == 0);
    CHECK(fmi3FreeFMUState(NULL, &st) == fmi3OK && st == NULL);
    CHECK(fmi3FreeFMUState(NULL, &st2) == fmi3OK);
    CHECK(fmi3FreeFMUState(NULL, &st) == fmi3OK);                    /* double free is a no-op */
    /* missing / unterminated state field */
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3Error);
    });
    WITH_SERVER("{\"ok\":true,\"state\":\"unterminated", 0, {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3Error);
    });
    Instance *dead = fake_instance(SOCK_INVALID);
    CHECK(fmi3SetFMUState((fmi3Instance)dead, NULL) == fmi3Error);
    CHECK(fmi3SerializedFMUStateSize(NULL, NULL, &n) == fmi3Error);
    /* importer-supplied bytes that are not base64 never reach the wire */
    fmi3FMUState junk = NULL;
    CHECK(fmi3DeserializeFMUState(NULL, (const fmi3Byte *)"abc\"def\\x", 9, &junk) == fmi3OK);
    g_log_calls = 0;
    CHECK(fmi3SetFMUState((fmi3Instance)dead, junk) == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "not valid") != NULL);
    fmi3FreeFMUState(NULL, &junk);
    free_instance(dead);
}

/* ------------------------------------------------ instantiate via TCP */

typedef struct { int listen_fd; const char *hello_reply; int accepted; } Listener;
static void *listener_thread(void *p) {
    Listener *L = (Listener *)p;
    int c = accept(L->listen_fd, NULL, NULL);
    if (c < 0) return NULL;
    L->accepted = 1;
    char *req = fake_exchange(c, L->hello_reply, 0);      /* hello */
    free(req);
    req = fake_exchange(c, "{\"ok\":true}", 0);          /* terminate (from FreeInstance) */
    free(req);
    sock_close(c);
    return NULL;
}

static void test_instantiate(const char *good_token) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr; memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); addr.sin_port = 0;
    CHECK(bind(fd, (struct sockaddr *)&addr, sizeof addr) == 0);
    socklen_t alen = sizeof addr;
    getsockname(fd, (struct sockaddr *)&addr, &alen);
    CHECK(listen(fd, 2) == 0);
    char ep[64]; snprintf(ep, sizeof ep, "127.0.0.1:%d", ntohs(addr.sin_port));
    setenv("MADDENING_FMU_ENDPOINT", ep, 1);

    char reply[256]; snprintf(reply, sizeof reply, "{\"ok\":true,\"token\":\"%s\",\"model\":\"m\"}", good_token);
    Listener L = { fd, reply, 0 };
    pthread_t th; pthread_create(&th, NULL, listener_thread, &L);
    fmi3Instance inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3True,
                                                    fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst != NULL);
    if (inst) {
        Instance *in = (Instance *)inst;
        CHECK(strcmp(in->instance_name, "i") == 0 && in->logging_on == fmi3True);
        CHECK(fmi3EnterInitializationMode(inst, fmi3False, 0, 0.5, fmi3False, 0) == fmi3OK && in->time == 0.5);
        CHECK(fmi3ExitInitializationMode(inst) == fmi3OK);
        fmi3FreeInstance(inst);                          /* sends terminate */
    }
    pthread_join(th, NULL);
    CHECK(L.accepted == 1);

    /* token mismatch -> NULL, with a log line */
    Listener L2 = { fd, reply, 0 };
    pthread_create(&th, NULL, listener_thread, &L2);
    g_log_calls = 0;
    inst = fmi3InstantiateCoSimulation("i", "wrong-token", NULL, fmi3False, fmi3False, fmi3False,
                                       fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst == NULL && g_log_calls >= 1 && strstr(g_last_log, "token") != NULL);
    shutdown(fd, SHUT_RDWR); sock_close(fd);
    pthread_join(th, NULL);

    /* no endpoint at all / closed port */
    unsetenv("MADDENING_FMU_ENDPOINT");
    g_log_calls = 0;
    CHECK(fmi3InstantiateCoSimulation("i", "t", "/nonexistent", fmi3False, fmi3False, fmi3False,
                                      fmi3False, NULL, 0, NULL, test_logger, NULL) == NULL);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "no endpoint") != NULL);
    setenv("MADDENING_FMU_ENDPOINT", ep, 1);          /* listener is gone now */
    CHECK(fmi3InstantiateCoSimulation("i", "t", NULL, fmi3False, fmi3False, fmi3False,
                                      fmi3False, NULL, 0, NULL, test_logger, NULL) == NULL);
    unsetenv("MADDENING_FMU_ENDPOINT");
    /* the other interface types refuse */
    CHECK(fmi3InstantiateModelExchange("i", "t", NULL, fmi3False, fmi3False, NULL, test_logger) == NULL);
    CHECK(fmi3InstantiateScheduledExecution("i", "t", NULL, fmi3False, fmi3False, NULL, test_logger,
                                            NULL, NULL, NULL) == NULL);
    fmi3FreeInstance(NULL);                              /* tolerated */
}

/* ------------------------------------------------- misc entry points */

static void test_misc_entry_points(void) {
    CHECK(strcmp(fmi3GetVersion(), "3.0") == 0);
    size_t n = 99;
    CHECK(fmi3GetNumberOfEventIndicators(NULL, &n) == fmi3OK && n == 0);
    CHECK(fmi3GetNumberOfContinuousStates(NULL, &n) == fmi3OK && n == 0);
    fmi3ValueReference vr[2] = { 1, 2 };
    fmi3Clock clk[2] = { fmi3ClockActive, fmi3ClockActive };
    CHECK(fmi3GetClock(NULL, vr, 2, clk) == fmi3OK && clk[0] == fmi3ClockInactive);
    CHECK(fmi3SetClock(NULL, vr, 2, clk) == fmi3OK);
    fmi3Boolean b1, b2, b3, b4, b5; fmi3Float64 t;
    CHECK(fmi3UpdateDiscreteStates(NULL, &b1, &b2, &b3, &b4, &b5, &t) == fmi3OK && !b1 && !b5);
    CHECK(fmi3EnterContinuousTimeMode(NULL) == fmi3Error);
    CHECK(fmi3GetString(NULL, vr, 1, NULL, 1) == fmi3Error);
    Instance *dead = fake_instance(SOCK_INVALID);
    CHECK(fmi3SetTime((fmi3Instance)dead, 3.0) == fmi3OK && dead->time == 3.0);
    CHECK(fmi3SetDebugLogging((fmi3Instance)dead, fmi3True, 0, NULL) == fmi3OK && dead->logging_on);
    CHECK(fmi3Terminate(NULL) == fmi3Error);
    CHECK(fmi3Reset(NULL) == fmi3Error);
    free_instance(dead);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    test_parse_values();
    test_read_endpoint();
    test_bridge_call_paths();
    test_get_set_step();
    test_fmu_state();
    test_instantiate("deadbeef-0000-4000-8000-000000000001");
    test_misc_entry_points();
    printf("maddening_fmu unit tests: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
