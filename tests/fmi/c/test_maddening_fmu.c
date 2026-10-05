/*
 * Unit tests for the FMU C wrapper.  Includes the wrapper source so the
 * static helpers (framing, JSON number parsing, endpoint discovery) are
 * testable directly; the FMI entry points are exercised against a fake
 * sidecar on a socketpair and, for instantiation, a real loopback
 * listener on a thread.  Built and run by tests/fmi/test_c_unit.py,
 * plain and under -fsanitize=address,undefined.
 *
 * WHICH SOURCE IS TESTED.  The wrapper is included through the macro
 * MADDENING_FMU_C, which tests/fmi/test_c_unit.py sets to the path of the
 * imported package's source (maddening.fmi.package.C_SOURCE), so the
 * binary tests the wrapper Python would package -- a copy of src/ put on
 * PYTHONPATH (a mutation harness's) included.  Built by hand without the
 * macro, the tree's own source is used.  main() prints the path.
 */

/* send() is routed through a fault injector for the send-failure tests
 * (test_send_failures).  Feature macros and the socket header come first,
 * so the macro below renames only the wrapper's call, never the
 * declaration in <sys/socket.h>. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE 1
#endif
#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>

/* Every allocation the wrapper and this file make is counted (the libc
 * headers come first, so the macros below rename only the calls).  The
 * count of live ones is what the FMU-state tests assert on -- a state
 * object allocated over one the importer handed back shows up as two more
 * -- and main() fails unless it ends at zero, so the plain build sees a
 * leak as the sanitized one does.  g_fail_allocs_in makes the n-th
 * allocation from now on this thread fail, once. */
static atomic_long g_live_allocs = 0;
static _Thread_local long g_fail_allocs_in = -1;
static int alloc_fails_now(void) {
    if (g_fail_allocs_in < 0) return 0;
    return g_fail_allocs_in-- == 0;
}
static void *counted_malloc(size_t n) {
    void *p = alloc_fails_now() ? NULL : malloc(n);
    if (p) atomic_fetch_add(&g_live_allocs, 1);
    return p;
}
static void *counted_calloc(size_t k, size_t n) {
    void *p = alloc_fails_now() ? NULL : calloc(k, n);
    if (p) atomic_fetch_add(&g_live_allocs, 1);
    return p;
}
static void *counted_realloc(void *old, size_t n) {
    void *p = alloc_fails_now() ? NULL : realloc(old, n);
    if (p && !old) atomic_fetch_add(&g_live_allocs, 1);
    return p;
}
static void counted_free(void *p) {
    if (p) atomic_fetch_sub(&g_live_allocs, 1);
    free(p);
}
static char *counted_strdup(const char *s) {
    char *p = strdup(s);
    if (p) atomic_fetch_add(&g_live_allocs, 1);
    return p;
}
static long live_allocs(void) { return atomic_load(&g_live_allocs); }
#define malloc counted_malloc
#define calloc counted_calloc
#define realloc counted_realloc
#define free counted_free
#define strdup counted_strdup

/* Faults apply to one socket only (the client end under test), never to
 * the fake server's sends on the other end of the pair. */
enum { FAULT_NONE = 0, FAULT_EINTR_ONCE, FAULT_HALF_THEN_RESET, FAULT_HALF_THEN_EINTR };
static int g_fault_fd = -1, g_fault_mode = FAULT_NONE, g_fault_calls = 0;
static ssize_t fault_send(int s, const void *buf, size_t n, int flags);
/* recv() likewise, for a signal that interrupts a receive (EINTR once, on
 * the socket in g_recv_fault_fd). */
static int g_recv_fault_fd = -1, g_recv_eintr = 0;
static ssize_t fault_recv(int s, void *buf, size_t n, int flags);
#define send fault_send
#define recv fault_recv
#ifndef MADDENING_FMU_C
#  define MADDENING_FMU_C "../../../src/maddening/fmi/c/maddening_fmu.c"
#endif
#include MADDENING_FMU_C
#undef send
#undef recv

static ssize_t fault_recv(int s, void *buf, size_t n, int flags) {
    if (s == g_recv_fault_fd && g_recv_eintr > 0) { --g_recv_eintr; errno = EINTR; return -1; }
    return recv(s, buf, n, flags);
}

static ssize_t fault_send(int s, const void *buf, size_t n, int flags) {
    if (s != g_fault_fd || g_fault_mode == FAULT_NONE) return send(s, buf, n, flags);
    int k = g_fault_calls++;
    switch (g_fault_mode) {
    case FAULT_EINTR_ONCE:                      /* a signal before anything is written */
        if (k == 0) { errno = EINTR; return -1; }
        break;
    case FAULT_HALF_THEN_RESET:                 /* a partial write, then a real failure */
        if (k == 0 && n > 1) return send(s, buf, n / 2, flags);
        if (k >= 1) { errno = ECONNRESET; return -1; }
        break;
    case FAULT_HALF_THEN_EINTR:                 /* a partial write, then a signal */
        if (k == 0 && n > 1) return send(s, buf, n / 2, flags);
        if (k == 1) { errno = EINTR; return -1; }
        break;
    default: break;
    }
    return send(s, buf, n, flags);
}

#include <assert.h>
#include <pthread.h>
#include <stddef.h>
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

/* In Step Mode, where get, set and doStep are allowed (the wrapper holds
 * FMI 3.0's state machine); a test of another state sets in->phase. */
static Instance *fake_instance(sock_t s) {
    Instance *in = (Instance *)calloc(1, sizeof *in);
    in->sock = s; in->log = test_logger;
    in->phase = PHASE_STEP_MODE;
    return in;
}

/* A state object of the test's own (on the stack, in no instance's list):
 * `n` bytes at `blob`, which fmi3SetFMUState only reads. */
static FmuState bare_state(char *blob, size_t n) {
    FmuState st;
    memset(&st, 0, sizeof st);
    st.blob = blob; st.n = n;
    return st;
}

/* The tests close the sockets themselves; this frees the rest (the
 * instance's C numeric locale and its live FMU states included). */
static void free_instance(Instance *in) {
    in->sock = SOCK_INVALID;
    instance_release(in);
}

/* Fake sidecar: reads one framed request from `s` (the flag bit and the
 * exact length are recorded, the body may hold NULs), then writes one
 * reply frame back: `advertised` bytes announced, `reply_len` bytes
 * actually sent, bit 31 set when `flag`.  reply == NULL closes the
 * socket without answering; a short frame hangs up after sending. */
typedef struct {
    sock_t s; const char *reply; size_t reply_len; size_t advertised; int flag;
    char *seen; size_t seen_len; int seen_flag; int closed;
} ServerArgs;

static void fake_exchange_x(ServerArgs *a) {
    unsigned char head[4];
    a->seen = NULL; a->seen_len = 0; a->seen_flag = 0; a->closed = 0;
    if (recv_all(a->s, (char *)head, 4)) return;
    unsigned long w = get_be32(head);
    size_t n = (size_t)(w & FRAME_LEN_MASK);
    a->seen_flag = (w & FRAME_BINARY) != 0;
    char *req = (char *)malloc(n + 1);
    if (recv_all(a->s, req, n)) { free(req); return; }
    req[n] = '\0';
    a->seen = req; a->seen_len = n;
    if (a->reply == NULL) { sock_close(a->s); a->closed = 1; return; }
    unsigned char h2[4];
    put_be32(h2, (unsigned long)a->advertised | (a->flag ? FRAME_BINARY : 0ul));
    send_all(a->s, (const char *)h2, 4);
    send_all(a->s, a->reply, a->reply_len);   /* may be shorter than advertised */
    if (a->advertised != a->reply_len) { sock_close(a->s); a->closed = 1; }
}

static size_t slen(const char *s) { return s ? strlen(s) : 0; }

/* JSON-only convenience used by the loopback listener. */
static char *fake_exchange(sock_t s, const char *reply, size_t reply_len_override) {
    size_t len = slen(reply);
    ServerArgs a = { s, reply, len, reply_len_override ? reply_len_override : len, 0,
                     NULL, 0, 0, 0 };
    fake_exchange_x(&a);
    return a.seen;
}

static void *server_thread(void *p) {
    fake_exchange_x((ServerArgs *)p);
    return NULL;
}

/* One exchange of the existing instance `in_` with a fake server that
 * answers once: `body` runs on the client end, and the request the server
 * saw is left in g_seen (g_seen_len bytes, flag bit in g_seen_flag) for the
 * caller to inspect.  The instance outlives the exchange, so a test can
 * make several on one instance (FMU states belong to their instance). */
static char *g_seen = NULL;
static size_t g_seen_len = 0;
static int g_seen_flag = 0;
#define EXCHANGE_X(in_, reply_, reply_len_, advertised_, flag_, body) do {       \
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);              \
    ServerArgs args = { sv[1], (reply_), (reply_len_), (advertised_), (flag_),    \
                        NULL, 0, 0, 0 };                                         \
    pthread_t th; pthread_create(&th, NULL, server_thread, &args);               \
    (in_)->sock = sv[0];                                                         \
    body;                                                                        \
    pthread_join(th, NULL);                                                      \
    if ((in_)->sock != SOCK_INVALID) sock_close(sv[0]);  /* unless the wrapper dropped it */ \
    (in_)->sock = SOCK_INVALID;                                                  \
    if (!args.closed) sock_close(sv[1]);                                         \
    free(g_seen); g_seen = args.seen ? args.seen : strdup("");                   \
    g_seen_len = args.seen_len; g_seen_flag = args.seen_flag;                    \
} while (0)
/* A JSON reply of strlen(reply) bytes. */
#define EXCHANGE(in_, reply, body) EXCHANGE_X((in_), (reply), slen(reply), slen(reply), 0, body)
/* A binary-flagged reply of `len` bytes. */
#define BINARY_EXCHANGE(in_, reply, len, body) EXCHANGE_X((in_), (reply), (len), (len), 1, body)

/* The same with an instance of its own, `in`, made for the exchange and
 * freed after it. */
#define WITH_SERVER_X(reply_, reply_len_, advertised_, flag_, body) do {         \
    Instance *in = fake_instance(SOCK_INVALID);                                  \
    EXCHANGE_X(in, (reply_), (reply_len_), (advertised_), (flag_), body);        \
    free_instance(in);                                                           \
} while (0)

/* A call the wrapper must refuse BEFORE it sends anything, made against a
 * server that would answer "ok" (with a value, a state and a time) to
 * whatever it was sent.  On a closed connection every call is fmi3Error,
 * so `call == fmi3Error` there could not fail and only a log-text check
 * could tell a refusal from a send; here an unrefused call is answered
 * fmi3OK, and the server must have seen nothing.  The instance `in` is in
 * Step Mode; `body` may set its phase or `in->binary`. */
#define WILLING_REPLY "{\"ok\":true,\"values\":[2],\"state\":\"QUJD\",\"t\":0}"
static int g_sent_nothing = 0;
#define WITH_WILLING_SERVER(body) do {                                           \
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);              \
    ServerArgs args = { sv[1], WILLING_REPLY, slen(WILLING_REPLY),               \
                        slen(WILLING_REPLY), 0, NULL, 0, 0, 0 };                 \
    pthread_t th; pthread_create(&th, NULL, server_thread, &args);               \
    Instance *in = fake_instance(sv[0]);                                         \
    body;                                                                        \
    /* hang up, so a server still waiting for a request sees EOF and ends */     \
    if (in->sock != SOCK_INVALID) sock_close(sv[0]);                             \
    pthread_join(th, NULL);                                                      \
    if (!args.closed) sock_close(sv[1]);                                         \
    g_sent_nothing = (args.seen == NULL);                                        \
    CHECK(g_sent_nothing);                       /* nothing reached the wire */  \
    free(args.seen);                                                             \
    free_instance(in);                                                           \
} while (0)

/* JSON reply of strlen(reply) bytes, optionally announcing len_override. */
#define WITH_SERVER(reply, len_override, body)                                   \
    WITH_SERVER_X((reply), slen(reply), (len_override) ? (len_override) : slen(reply), 0, body)
/* Binary-flagged reply of `len` bytes (announced as is). */
#define WITH_BINARY_SERVER(reply, len, body) WITH_SERVER_X((reply), (len), (len), 1, body)

/* [u32 BE header_len][hdr][raw] into out; returns the payload length. */
static size_t bin_payload(unsigned char *out, const char *hdr, const void *raw, size_t rawlen) {
    size_t hl = strlen(hdr);
    put_be32(out, (unsigned long)hl);
    memcpy(out + 4, hdr, hl);
    if (rawlen) memcpy(out + 4 + hl, raw, rawlen);
    return 4 + hl + rawlen;
}

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
    /* more values than nValues: refused, not the first n with the rest
     * dropped (FMI's nValues is what the value references hold) */
    g_log_calls = 0;
    CHECK(parse_values(in, out, 3) == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "more values in reply than nValues") != NULL);
    CHECK(parse_values(in, out, 1) == fmi3Error);
    free(in->resp);
    /* a list that does not close where the n-th value ends is malformed */
    in->resp = strdup("{\"ok\":true,\"values\":[1,2}");
    CHECK(parse_values(in, out, 2) == fmi3Error);
    CHECK(strstr(g_last_log, "malformed values list") != NULL);
    free(in->resp);
    in->resp = strdup("{\"ok\":true,\"values\":[1, 2 ]}");
    CHECK(parse_values(in, out, 2) == fmi3OK && out[0] == 1.0 && out[1] == 2.0);
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

/* -------------------------------------------- parse_values, non-finite
 *
 * MADD-ANO-006: the bridge writes a non-finite value as a *quoted* token,
 * because the bare tokens json.dumps used to emit are not JSON.  C99
 * strtod parses the token itself; the quote is what it cannot step over.
 * Both spellings must read, and a quoted field must be properly closed.
 */

static void test_parse_values_non_finite(void) {
    Instance *in = fake_instance(SOCK_INVALID);
    double out[4];

    /* the quoted form the bridge writes since 0.4.0 */
    in->resp = strdup("{\"ok\":true,\"values\":[\"NaN\",\"Infinity\",\"-Infinity\",2.5]}");
    CHECK(parse_values(in, out, 4) == fmi3OK);
    CHECK(isnan(out[0]));
    CHECK(isinf(out[1]) && out[1] > 0);
    CHECK(isinf(out[2]) && out[2] < 0);
    CHECK(out[3] == 2.5);
    free(in->resp);

    /* the bare form a pre-0.4.0 bridge writes: still accepted */
    in->resp = strdup("{\"ok\":true,\"values\":[NaN,Infinity,-Infinity,2.5]}");
    CHECK(parse_values(in, out, 4) == fmi3OK);
    CHECK(isnan(out[0]));
    CHECK(isinf(out[1]) && out[1] > 0);
    CHECK(isinf(out[2]) && out[2] < 0);
    CHECK(out[3] == 2.5);
    free(in->resp);

    /* a quoted ordinary number reads too -- the rule is about quotes,
     * not about which tokens are inside them */
    in->resp = strdup("{\"ok\":true,\"values\":[\"1.5\", \"-2e3\"]}");
    CHECK(parse_values(in, out, 2) == fmi3OK);
    CHECK(out[0] == 1.5 && out[1] == -2000.0);
    free(in->resp);

    /* an unterminated quoted field is malformed, not silently truncated:
     * without the closing-quote check "1e5xyz" would pass as 1e5 */
    in->resp = strdup("{\"ok\":true,\"values\":[\"1e5xyz\"]}");
    CHECK(parse_values(in, out, 1) == fmi3Error);
    free(in->resp);
    /* ... and one junk character before the list closes: only the
     * closing-quote check refuses this ("1e5xyz" is refused again by the
     * check that the list ends after n values, so it alone could not tell
     * whether the closing-quote check exists) */
    g_log_calls = 0;
    in->resp = strdup("{\"ok\":true,\"values\":[\"1e5x]}");
    CHECK(parse_values(in, out, 1) == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "malformed quoted number") != NULL);
    free(in->resp);

    /* a bare quote with nothing parseable after it is still malformed */
    in->resp = strdup("{\"ok\":true,\"values\":[\"\"]}");
    CHECK(parse_values(in, out, 1) == fmi3Error);
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
    /* server closes without replying: the connection is dead from now on */
    WITH_SERVER(NULL, 0, {
        CHECK(bridge_call(in, "{\"op\":\"x\"}") == fmi3Error);
        CHECK(in->sock == SOCK_INVALID);
    });
    /* header advertises more bytes than sent: recv fails, no read past
     * buffer, and the half-read stream is not reused */
    WITH_SERVER("{\"ok\":true}", 4096, {
        CHECK(bridge_call(in, "{\"op\":\"x\"}") == fmi3Error);
        CHECK(in->sock == SOCK_INVALID);
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
    g_log_calls = 0;
    CHECK(bridge_call(dead, "{}") == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "connection to the sidecar is closed") != NULL);
    free_instance(dead);
    /* the peer is gone before we send: EPIPE -> fmi3Error, not SIGPIPE;
     * and the connection is dropped, as on a failed receive */
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    sock_close(sv[1]);
    Instance *gone = fake_instance(sv[0]);
    g_log_calls = 0;
    CHECK(bridge_call(gone, "{\"op\":\"hello\"}") == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "send failed") != NULL);
    CHECK(gone->sock == SOCK_INVALID);
    if (gone->sock != SOCK_INVALID) sock_close(sv[0]);
    free_instance(gone);
}

/* ------------------------------------------------------- recv: EINTR */

static void test_a_signal_during_a_receive_is_retried(void) {
    /* A signal that interrupts recv() before anything arrives is not a
     * failure: the receive is retried and the reply read whole (recv_all
     * used to treat it as a dead peer and drop the connection). */
    WITH_SERVER("{\"ok\":true,\"t\":1}", 0, {
        g_recv_fault_fd = in->sock; g_recv_eintr = 2;
        CHECK(bridge_call(in, "{\"op\":\"hello\"}") == fmi3OK);
        CHECK(g_recv_eintr == 0 && in->sock != SOCK_INVALID);
        CHECK(strcmp(in->resp, "{\"ok\":true,\"t\":1}") == 0);
        g_recv_fault_fd = -1;
    });
}

/* ------------------------------------------------------- send failures */

static void test_send_failures(void) {
    /* A signal before anything is written is retried: the request goes
     * out whole and the call succeeds. */
    WITH_SERVER("{\"ok\":true}", 0, {
        g_fault_fd = in->sock; g_fault_mode = FAULT_EINTR_ONCE; g_fault_calls = 0;
        CHECK(bridge_call(in, "{\"op\":\"terminate\"}") == fmi3OK);
        CHECK(in->sock != SOCK_INVALID);
        g_fault_mode = FAULT_NONE;
    });
    CHECK(strcmp(g_seen, "{\"op\":\"terminate\"}") == 0);
    /* A partial write and then a signal: the rest is sent, the frame
     * arrives intact, the call succeeds (send_all used to give up on
     * EINTR with half a frame on the wire). */
    WITH_SERVER("{\"ok\":true}", 0, {
        g_fault_fd = in->sock; g_fault_mode = FAULT_HALF_THEN_EINTR; g_fault_calls = 0;
        CHECK(bridge_call(in, "{\"op\":\"get\",\"vr\":[7]}") == fmi3OK);
        CHECK(in->sock != SOCK_INVALID);
        g_fault_mode = FAULT_NONE;
    });
    CHECK(strcmp(g_seen, "{\"op\":\"get\",\"vr\":[7]}") == 0);
    /* A partial write and then a real failure: the stream is out of step
     * (the peer holds half a frame and would read the next request as its
     * rest), so the connection is dropped -- every later call fails
     * honestly, and none is ever read as the remainder. */
    WITH_SERVER("{\"ok\":true}", 0, {
        g_fault_fd = in->sock; g_fault_mode = FAULT_HALF_THEN_RESET; g_fault_calls = 0;
        g_log_calls = 0;
        CHECK(bridge_call(in, "{\"op\":\"get\",\"vr\":[7]}") == fmi3Error);
        CHECK(g_log_calls == 1 && strstr(g_last_log, "send failed") != NULL);
        CHECK(in->sock == SOCK_INVALID);
        g_fault_mode = FAULT_NONE; g_fault_fd = -1;
        CHECK(bridge_call(in, "{\"op\":\"get\",\"vr\":[7]}") == fmi3Error);
        CHECK(strstr(g_last_log, "connection to the sidecar is closed") != NULL);
    });
    g_fault_mode = FAULT_NONE; g_fault_fd = -1;
}

/* --------------------------------------------------- get / set / step */

static void test_get_set_step(void) {
    fmi3ValueReference vr[3] = { 7, 9, 11 };
    fmi3Float64 v64[3] = { 1.5, -2.0, 1e-9 };
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetFloat64((fmi3Instance)in, vr, 3, v64, 3) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"set\",\"type\":\"Float64\",\"vr\":[7,9,11],\"values\":[1.5,-2,1.0000000000000001e-09]}") == 0);
    fmi3Int32 vi[2] = { -5, 7 };
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetInt32((fmi3Instance)in, vr, 2, vi, 2) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"set\",\"type\":\"Int32\",\"vr\":[7,9],\"values\":[-5,7]}") == 0);
    fmi3Boolean vb[1] = { fmi3True };
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetBoolean((fmi3Instance)in, vr, 1, vb, 1) == fmi3OK);
    });
    /* every request names the type of the function called, so the bridge
     * can refuse fmi3SetBoolean on a Float32 variable (it used to arrive
     * as a bare 1.0 and be stored) */
    CHECK(strcmp(g_seen, "{\"op\":\"set\",\"type\":\"Boolean\",\"vr\":[7],\"values\":[1]}") == 0);
    /* NaN / inf never leave the process: refused with a peer that would
     * have taken them */
    fmi3Float32 bad[1] = { NAN };
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        CHECK(fmi3SetFloat32((fmi3Instance)in, vr, 1, bad, 1) == fmi3Error);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "non-finite") != NULL);
    /* nValues == 0 needs no socket */
    Instance *dead = fake_instance(SOCK_INVALID);
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
    CHECK(strcmp(g_seen, "{\"op\":\"get\",\"type\":\"Float32\",\"vr\":[7,9]}") == 0);
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
    /* and with more values than nValues: error too, outputs untouched */
    WITH_SERVER("{\"ok\":true,\"values\":[1,2,3]}", 0, {
        CHECK(fmi3GetFloat32((fmi3Instance)in, vr, 2, g32, 2) == fmi3Error);
    });
    CHECK(g32[0] == 42.0f);
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
    /* a non-finite point or step never reaches the wire (%.17g would
     * write "nan", which is not JSON) */
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        in->time = 0.3;
        CHECK(fmi3DoStep((fmi3Instance)in, NAN, 0.05, fmi3False, &ev, &term, &early, &last) == fmi3Error);
        CHECK(last == 0.3 && in->time == 0.3);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "must be finite") != NULL);
    WITH_WILLING_SERVER({
        in->time = 0.3;
        CHECK(fmi3DoStep((fmi3Instance)in, 0.3, INFINITY, fmi3False, &ev, &term, &early, &last) == fmi3Error);
        CHECK(last == 0.3 && in->time == 0.3);
    });
    /* EnterInitializationMode tells the bridge the start time, which is the
     * instance's time from then on (it used to stay in the wrapper, and the
     * "time" variable read 0.0 until the first step) */
    WITH_SERVER("{\"ok\":true,\"t\":5}", 0, {
        in->phase = PHASE_INSTANTIATED;
        CHECK(fmi3EnterInitializationMode((fmi3Instance)in, fmi3False, 0, 5.0, fmi3False, 0) == fmi3OK);
        CHECK(in->time == 5.0 && in->phase == PHASE_INITIALIZATION_MODE);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"initialize\",\"t\":5}") == 0);
    WITH_SERVER("{\"ok\":false,\"error\":\"ValueError: initialize after the instance has stepped\"}", 0, {
        in->time = 0.7; in->phase = PHASE_INSTANTIATED;
        g_log_calls = 0;
        CHECK(fmi3EnterInitializationMode((fmi3Instance)in, fmi3False, 0, 0.0, fmi3False, 0) == fmi3Error);
        CHECK(in->time == 0.7 && strstr(g_last_log, "has stepped") != NULL);
        CHECK(in->phase == PHASE_INSTANTIATED);          /* a refused initialize moves nothing */
    });
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        in->phase = PHASE_INSTANTIATED;
        CHECK(fmi3EnterInitializationMode((fmi3Instance)in, fmi3False, 0, NAN, fmi3False, 0) == fmi3Error);
        CHECK(in->phase == PHASE_INSTANTIATED);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "start time must be finite") != NULL);
    CHECK(fmi3EnterInitializationMode(NULL, fmi3False, 0, 0.0, fmi3False, 0) == fmi3Error);
}

/* ------------------------------------------------------- binary frames */

static void test_binary_get_set(void) {
    fmi3ValueReference vr[3] = { 7, 9, 11 };
    unsigned char frame[256];
    double two[2] = { 2.5, -1e-300 };
    unsigned char le[16];
    f64_to_le(le, two, 2);
    /* a binary get reply carries the doubles verbatim (no strtod) */
    size_t n = bin_payload(frame, "{\"ok\":true,\"n\":2,\"dtype\":\"f64\"}", le, 16);
    fmi3Float64 g64[2] = { 0, 0 };
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3OK);
        CHECK(g64[0] == 2.5 && g64[1] == -1e-300);
        CHECK(in->resp_binary && in->raw_len == 16 && strcmp(in->hdr, "{\"ok\":true,\"n\":2,\"dtype\":\"f64\"}") == 0);
    });
    CHECK(g_seen_flag == 0 && strcmp(g_seen, "{\"op\":\"get\",\"type\":\"Float64\",\"vr\":[7,9]}") == 0);  /* get stays JSON */
    /* narrower widths are widened from the same float64 wire form */
    fmi3Float32 g32[2];
    WITH_BINARY_SERVER((const char *)frame, n, {
        CHECK(fmi3GetFloat32((fmi3Instance)in, vr, 2, g32, 2) == fmi3OK);   /* parsed by flag, not mode */
        CHECK(g32[0] == 2.5f);
    });
    /* fewer values requested than sent is an error, as more is: nValues
     * is the number of values the value references hold, so a reply of
     * any other length answers a different request (it used to return
     * fmi3OK with the extra values dropped) */
    g64[0] = 42;
    g_log_calls = 0;
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 1, g64, 1) == fmi3Error);
    });
    CHECK(g64[0] == 42 && strstr(g_last_log, "more values in reply than nValues") != NULL);
    fmi3Float64 g3[3] = { 42, 42, 42 };
    g_log_calls = 0;
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 3, g3, 3) == fmi3Error);
    });
    CHECK(g3[0] == 42 && strstr(g_last_log, "too few") != NULL);
    /* header count and raw length disagree (both directions) */
    n = bin_payload(frame, "{\"ok\":true,\"n\":3,\"dtype\":\"f64\"}", le, 16);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    CHECK(strstr(g_last_log, "length mismatch") != NULL);
    n = bin_payload(frame, "{\"ok\":true,\"n\":2,\"dtype\":\"f64\"}", le, 15);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    /* n far too large for the raw part / for any frame, negative, absent */
    n = bin_payload(frame, "{\"ok\":true,\"n\":99999999999999999999,\"dtype\":\"f64\"}", le, 16);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    n = bin_payload(frame, "{\"ok\":true,\"n\":1099511627776,\"dtype\":\"f64\"}", le, 16);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    n = bin_payload(frame, "{\"ok\":true,\"n\":-2,\"dtype\":\"f64\"}", le, 16);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    n = bin_payload(frame, "{\"ok\":true,\"dtype\":\"f64\"}", le, 16);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    /* a count that is not a plain decimal ("2abc", "1e3", "0x10") is no count */
    n = bin_payload(frame, "{\"ok\":true,\"n\":2abc,\"dtype\":\"f64\"}", le, 16);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    CHECK(strstr(g_last_log, "no valid count") != NULL);
    {
        Instance tmp; size_t c = 99;
        memset(&tmp, 0, sizeof tmp);
        snprintf(tmp.hdr, sizeof tmp.hdr, "{\"ok\":true,\"n\":1e3}");   CHECK(hdr_count(&tmp, &c) == -1);
        snprintf(tmp.hdr, sizeof tmp.hdr, "{\"ok\":true,\"n\":0x10}");  CHECK(hdr_count(&tmp, &c) == -1);
        snprintf(tmp.hdr, sizeof tmp.hdr, "{\"ok\":true,\"n\": 12 }");  CHECK(hdr_count(&tmp, &c) == 0 && c == 12);
        snprintf(tmp.hdr, sizeof tmp.hdr, "{\"ok\":true,\"n\":7,\"dtype\":\"f64\"}");
        CHECK(hdr_count(&tmp, &c) == 0 && c == 7);
    }
    /* wrong / missing dtype */
    n = bin_payload(frame, "{\"ok\":true,\"n\":2,\"dtype\":\"f32\"}", le, 16);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    CHECK(strstr(g_last_log, "float64") != NULL);
    /* truncated raw part: the announced length never arrives; the
     * connection is dropped (it can never be back in sync) */
    n = bin_payload(frame, "{\"ok\":true,\"n\":2,\"dtype\":\"f64\"}", le, 16);
    WITH_SERVER_X((const char *)frame, n - 5, n, 1, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
        CHECK(in->resp[0] == '\0' && !in->resp_binary);
        CHECK(in->sock == SOCK_INVALID);
    });
    /* header_len larger than the payload, or larger than the header cap:
     * the frame was read in full, so the connection stays usable */
    put_be32(frame, 1000);
    WITH_BINARY_SERVER((const char *)frame, 40, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
        CHECK(!in->resp_binary);
        CHECK(in->sock != SOCK_INVALID);
    });
    CHECK(strstr(g_last_log, "malformed") != NULL);
    {
        unsigned char *huge = (unsigned char *)malloc(HDR_MAX + 64);
        put_be32(huge, HDR_MAX);
        memset(huge + 4, ' ', HDR_MAX + 60);
        WITH_BINARY_SERVER((const char *)huge, HDR_MAX + 64, {
            in->binary = 1;
            CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
        });
        free(huge);
    }
    /* a frame shorter than the header-length field */
    WITH_BINARY_SERVER("ab", 2, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    /* binary flag with an oversize length: refused before any allocation */
    WITH_SERVER_X("x", 1, (size_t)FRAME_LEN_MASK, 1, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
        CHECK(in->resp_cap < 1000 && strstr(g_last_log, "frame limit") != NULL);
    });
    WITH_SERVER_X("x", 1, (size_t)FRAME_MAX + 1, 1, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
        CHECK(in->resp_cap < 1000);
    });
    /* exactly the limit is still read (the bridge may send that much): the
     * frame is taken in full and judged on its contents -- here refused for
     * its count, which is not the caller's nValues, with the connection
     * intact -- never dropped as over the limit */
    {
        unsigned char *max = (unsigned char *)malloc(FRAME_MAX);
        size_t hl = bin_payload(max, "{\"ok\":true,\"n\":8388602,\"dtype\":\"f64\",\"p\":12}", NULL, 0) - 4;
        memset(max + 4 + hl, 0, FRAME_MAX - 4 - hl);
        CHECK(4 + hl + 8 * 8388602ul == FRAME_MAX);
        g_log_calls = 0;
        WITH_BINARY_SERVER((const char *)max, FRAME_MAX, {
            in->binary = 1;
            CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
            CHECK(in->sock != SOCK_INVALID && in->resp_binary && in->raw_len == 8 * 8388602ul);
        });
        CHECK(g_log_calls == 1 && strstr(g_last_log, "more values in reply than nValues") != NULL);
        free(max);
    }
    /* a binary-flagged error reply is reported like a JSON one */
    n = bin_payload(frame, "{\"ok\":false,\"error\":\"KeyError: nope\"}", NULL, 0);
    g_log_calls = 0;
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3Error);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "KeyError: nope") != NULL);
    /* binary set: header + raw doubles, no %.17g */
    fmi3Float64 v64[3] = { 1.5, -2.0, 1e-9 };
    WITH_SERVER("{\"ok\":true}", 0, {
        in->binary = 1;
        CHECK(fmi3SetFloat64((fmi3Instance)in, vr, 3, v64, 3) == fmi3OK);
    });
    CHECK(g_seen_flag == 1);
    {
        const char *hdr = "{\"op\":\"set\",\"type\":\"Float64\",\"vr\":[7,9,11],\"n\":3,\"dtype\":\"f64\"}";
        size_t hl = strlen(hdr);
        CHECK(g_seen_len == 4 + hl + 24);
        CHECK(get_be32((const unsigned char *)g_seen) == hl);
        CHECK(memcmp(g_seen + 4, hdr, hl) == 0);
        double back[3];
        f64_from_le(back, (const unsigned char *)g_seen + 4 + hl, 3);
        CHECK(back[0] == 1.5 && back[1] == -2.0 && back[2] == 1e-9);
    }
    fmi3Int32 vi[2] = { -5, 7 };
    WITH_SERVER("{\"ok\":true}", 0, {
        in->binary = 1;
        CHECK(fmi3SetInt32((fmi3Instance)in, vr, 2, vi, 2) == fmi3OK);
    });
    CHECK(g_seen_flag == 1 && g_seen_len == 4 + strlen("{\"op\":\"set\",\"type\":\"Int32\",\"vr\":[7,9],\"n\":2,\"dtype\":\"f64\"}") + 16);
    /* non-finite values never leave the process on the binary path either */
    fmi3Float64 bad[2] = { 1.0, INFINITY };
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(fmi3SetFloat64((fmi3Instance)in, vr, 2, bad, 2) == fmi3Error);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "non-finite") != NULL);
    /* a set larger than the frame limit is refused before any buffer grows
     * (and before any value is read: the count alone decides) */
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(do_set(in, "Float64", vr, 1, v64, (size_t)FRAME_MAX / 8 + 1) == fmi3Error);
        CHECK(in->req_cap == 0);
    });
    CHECK(strstr(g_last_log, "frame limit") != NULL);
    /* endianness helpers round-trip on this host */
    {
        double x[2] = { 0.1, -1.7976931348623157e308 }, y[2];
        unsigned char w[16];
        f64_to_le(w, x, 2); f64_from_le(y, w, 2);
        CHECK(y[0] == 0.1 && y[1] == -1.7976931348623157e308);
        if (host_is_little_endian()) CHECK(memcmp(w, x, 16) == 0);
    }
}

static void test_binary_fmu_state(void) {
    /* a get_state reply carries the npz bytes raw: NULs and all */
    unsigned char blob[12] = { 'P', 'K', 0, 0, 1, 255, 0, 3, '"', '\\', 0, 9 };
    unsigned char frame[128];
    size_t n = bin_payload(frame, "{\"ok\":true,\"n\":12}", blob, 12);
    Instance *in = fake_instance(SOCK_INVALID);     /* the states below are its */
    in->binary = 1;
    fmi3FMUState st = NULL;
    BINARY_EXCHANGE(in, (const char *)frame, n, {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK);
    });
    CHECK(st != NULL && ((FmuState *)st)->n == 12 && memcmp(((FmuState *)st)->blob, blob, 12) == 0);
    /* serialize / deserialize keep the raw bytes */
    size_t sz = 0;
    CHECK(fmi3SerializedFMUStateSize((fmi3Instance)in, st, &sz) == fmi3OK && sz == 12);
    fmi3Byte buf[12];
    CHECK(fmi3SerializeFMUState((fmi3Instance)in, st, buf, 12) == fmi3OK && memcmp(buf, blob, 12) == 0);
    fmi3FMUState st2 = NULL;
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, buf, 12, &st2) == fmi3OK);
    /* set_state on the binary path is length-delimited: no base64 check */
    EXCHANGE(in, "{\"ok\":true}", {
        CHECK(fmi3SetFMUState((fmi3Instance)in, st2) == fmi3OK);
    });
    CHECK(g_seen_flag == 1);
    {
        const char *hdr = "{\"op\":\"set_state\",\"n\":12}";
        size_t hl = strlen(hdr);
        CHECK(g_seen_len == 4 + hl + 12 && get_be32((const unsigned char *)g_seen) == hl);
        CHECK(memcmp(g_seen + 4, hdr, hl) == 0 && memcmp(g_seen + 4 + hl, blob, 12) == 0);
    }
    /* the same blob on a JSON-mode instance is refused (not base64), by the
     * wrapper: the peer would have taken it */
    g_log_calls = 0;
    WITH_WILLING_SERVER({ CHECK(fmi3SetFMUState((fmi3Instance)in, st2) == fmi3Error); });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "not valid") != NULL);
    /* an oversize blob is refused before the request buffer grows */
    FmuState huge = bare_state((char *)"x", (size_t)FRAME_MAX + 1);
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(fmi3SetFMUState((fmi3Instance)in, &huge) == fmi3Error && in->req_cap == 0);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &st) == fmi3OK && st == NULL);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &st2) == fmi3OK && st2 == NULL);
    /* count / raw length mismatch on get_state */
    n = bin_payload(frame, "{\"ok\":true,\"n\":11}", blob, 12);
    BINARY_EXCHANGE(in, (const char *)frame, n, {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3Error);
    });
    CHECK(st == NULL && strstr(g_last_log, "length mismatch") != NULL);
    n = bin_payload(frame, "{\"ok\":true}", blob, 12);
    BINARY_EXCHANGE(in, (const char *)frame, n, {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3Error);
    });
    CHECK(st == NULL);
    /* a JSON get_state reply still works on a binary-mode instance */
    EXCHANGE(in, "{\"ok\":true,\"state\":\"QUJDRA==\"}", {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK);
    });
    CHECK(st != NULL && ((FmuState *)st)->n == 8);
    fmi3FreeFMUState((fmi3Instance)in, &st);
    free_instance(in);
}

/* ------------------------------------------- over-limit / broken framing */

static void test_oversize_reply_kills_the_connection(void) {
    /* A binary reply over the frame limit is refused unread.  What the
     * peer sent after the prefix is then still in the socket; before the
     * fix an unanswered DoStep parsed those stale bytes as its reply and
     * returned fmi3OK.  Now the connection is dropped: every later call
     * is fmi3Error, never a false success. */
    fmi3ValueReference vr[1] = { 7 }; fmi3Float64 out[1] = { 0 };
    unsigned char stale[4 + 18];
    put_be32(stale, 18);
    memcpy(stale + 4, "{\"ok\":true,\"t\":9}", 18);       /* a well-formed JSON frame */
    fmi3Boolean ev, term, early; fmi3Float64 last = -1;
    WITH_SERVER_X((const char *)stale, sizeof stale, (size_t)FRAME_MAX + 1, 1, {
        in->binary = 1;
        g_log_calls = 0;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 1, out, 1) == fmi3Error);
        CHECK(strstr(g_last_log, "frame limit") != NULL);
        CHECK(in->sock == SOCK_INVALID && in->resp_cap < 1000);
        in->time = 0.0;
        CHECK(fmi3DoStep((fmi3Instance)in, 0.0, 0.5, fmi3False, &ev, &term, &early, &last) == fmi3Error);
        CHECK(last == 0.0 && in->time == 0.0);
        CHECK(strstr(g_last_log, "connection to the sidecar is closed") != NULL);
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 1, out, 1) == fmi3Error);
    });
    /* the same for a JSON (unflagged) reply: it used to get a 2 GiB
     * realloc before the first body byte, now it is refused unallocated */
    WITH_SERVER_X((const char *)stale, sizeof stale, (size_t)FRAME_LEN_MASK, 0, {
        g_log_calls = 0;
        CHECK(bridge_call(in, "{\"op\":\"step\"}") == fmi3Error);
        CHECK(in->resp_cap < 1000 && strstr(g_last_log, "frame limit") != NULL);
        CHECK(in->sock == SOCK_INVALID);
        CHECK(fmi3DoStep((fmi3Instance)in, 0.0, 0.5, fmi3False, &ev, &term, &early, &last) == fmi3Error);
    });
    WITH_SERVER_X((const char *)stale, sizeof stale, (size_t)FRAME_MAX + 1, 0, {
        CHECK(bridge_call(in, "{\"op\":\"step\"}") == fmi3Error && in->sock == SOCK_INVALID);
    });
    /* FreeInstance on a dropped connection sends nothing and frees cleanly */
    {
        int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        Instance *in = fake_instance(sv[0]);
        conn_drop(in, "test");
        fmi3FreeInstance((fmi3Instance)in);
        char c; CHECK(recv(sv[1], &c, 1, 0) == 0);          /* EOF, no terminate frame */
        sock_close(sv[1]);
    }
}

static void test_max_set_frame_fits_the_bridge_limit(void) {
    /* The largest binary set the wrapper lets through is at most FRAME_MAX
     * bytes on the wire, header included; one more value, or a longer
     * value-reference list, is refused before anything is sent (the
     * bridge answers an over-limit frame by dropping the connection
     * without an error reply). */
    fmi3ValueReference vr[20];
    for (size_t i = 0; i < 20; ++i) vr[i] = 4294967295u;
    char hdr[128];
    size_t n = FRAME_MAX / 8, hl = 0;
    for (;;) {
        hl = (size_t)snprintf(hdr, sizeof hdr, "{\"op\":\"set\",\"type\":\"Float64\",\"vr\":[%u],\"n\":%lu,\"dtype\":\"f64\"}",
                              (unsigned)vr[0], (unsigned long)n);
        if (4 + hl + 8 * n <= FRAME_MAX) break;
        --n;
    }
    CHECK(4 + hl + 8 * (n + 1) > FRAME_MAX);              /* n is the largest that fits */
    double *vals = (double *)calloc(n + 1, sizeof(double));
    WITH_SERVER("{\"ok\":true}", 0, {
        in->binary = 1;
        CHECK(do_set(in, "Float64", vr, 1, vals, n) == fmi3OK);
    });
    CHECK(g_seen_flag == 1 && g_seen_len == 4 + hl + 8 * n && g_seen_len <= FRAME_MAX);
    CHECK(get_be32((const unsigned char *)g_seen) == hl && memcmp(g_seen + 4, hdr, hl) == 0);
    /* one value more is refused by the wrapper, with a peer that would
     * have answered it */
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(do_set(in, "Float64", vr, 1, vals, n + 1) == fmi3Error);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
    /* the header counts: twenty 10-digit value references push the same
     * payload over the limit even with a few values fewer */
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(do_set(in, "Float64", vr, 20, vals, n - 10) == fmi3Error);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
    /* and an absurd count or vr list is refused before any buffer grows */
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(do_set(in, "Float64", vr, 1, vals, (size_t)FRAME_MAX / 8 + 1) == fmi3Error && in->req_cap == 0);
    });
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(do_set(in, "Float64", vr, (size_t)FRAME_MAX, vals, 1) == fmi3Error && in->req_cap == 0);
    });
    free(vals);
}

static void test_get_request_respects_the_frame_limit(void) {
    /* do_set has always refused a request frame over FRAME_MAX; do_get had
     * no check at all, so a get of a few million value references built a
     * request the bridge refuses to read -- the connection is dropped and
     * the instance dies, instead of the importer being told the request is
     * too large.  (Audit params-io 2026-09-19, repro/c_probe.c: nvr =
     * 7,000,000 produced a 77,000,019-byte frame against a 67,108,864-byte
     * limit.)  The cheap bound is checked here; the exact post-build one
     * needs a 150 MB buffer, so it is left to the reproducer. */
    fmi3ValueReference vr[1] = { 4294967295u };
    double out[1];
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        CHECK(do_get(in, "Float64", vr, (size_t)FRAME_MAX / 2 + 1, out, 1) == fmi3Error);
        CHECK(in->req_cap == 0);                   /* refused before any buffer grew */
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
}


/* ------------------------------------- typed access: values a type holds */

static void test_getters_refuse_values_their_type_cannot_hold(void) {
    /* The bridge refuses a variable of another type (the request names
     * it); the wrapper still checks every reply value before converting
     * it, because (CTYPE)NaN, (int32)1e10 and friends are undefined
     * behaviour and (int32)0.5 is a silent 0.  values[] is untouched on a
     * refusal. */
    fmi3ValueReference vr[2] = { 7, 9 };
    fmi3Int32 i32[2] = { 42, 42 };
    const char *bad_i32[] = { "[0.5,1]", "[1,\"NaN\"]", "[2147483648,0]", "[-2147483649,0]",
                              "[1,\"Infinity\"]" };
    for (size_t k = 0; k < sizeof bad_i32 / sizeof *bad_i32; ++k) {
        char reply[96]; snprintf(reply, sizeof reply, "{\"ok\":true,\"values\":%s}", bad_i32[k]);
        g_log_calls = 0;
        WITH_SERVER(reply, 0, {
            CHECK(fmi3GetInt32((fmi3Instance)in, vr, 2, i32, 2) == fmi3Error);
            CHECK(in->sock != SOCK_INVALID);          /* a refused value, not a broken stream */
        });
        CHECK(i32[0] == 42 && i32[1] == 42);
        CHECK(g_log_calls == 1 && strstr(g_last_log, "not a value of type Int32") != NULL);
    }
    CHECK(strcmp(g_seen, "{\"op\":\"get\",\"type\":\"Int32\",\"vr\":[7,9]}") == 0);
    WITH_SERVER("{\"ok\":true,\"values\":[-2147483648,2147483647]}", 0, {
        CHECK(fmi3GetInt32((fmi3Instance)in, vr, 2, i32, 2) == fmi3OK);
    });
    CHECK(i32[0] == INT32_MIN && i32[1] == INT32_MAX);
    fmi3UInt8 u8[1] = { 7 };
    WITH_SERVER("{\"ok\":true,\"values\":[256]}", 0, {
        CHECK(fmi3GetUInt8((fmi3Instance)in, vr, 1, u8, 1) == fmi3Error);
    });
    WITH_SERVER("{\"ok\":true,\"values\":[-1]}", 0, {
        CHECK(fmi3GetUInt8((fmi3Instance)in, vr, 1, u8, 1) == fmi3Error);
    });
    CHECK(u8[0] == 7);
    fmi3UInt64 u64[1] = { 7 };
    WITH_SERVER("{\"ok\":true,\"values\":[18446744073709551616]}", 0, {   /* 2^64 */
        CHECK(fmi3GetUInt64((fmi3Instance)in, vr, 1, u64, 1) == fmi3Error);
    });
    WITH_SERVER("{\"ok\":true,\"values\":[18446744073709549568]}", 0, {   /* largest double < 2^64 */
        CHECK(fmi3GetUInt64((fmi3Instance)in, vr, 1, u64, 1) == fmi3OK);
    });
    CHECK(u64[0] == 18446744073709549568ull);
    fmi3Int64 i64[1] = { 7 };
    WITH_SERVER("{\"ok\":true,\"values\":[9223372036854775808]}", 0, {    /* 2^63 */
        CHECK(fmi3GetInt64((fmi3Instance)in, vr, 1, i64, 1) == fmi3Error);
    });
    WITH_SERVER("{\"ok\":true,\"values\":[-9223372036854775808]}", 0, {
        CHECK(fmi3GetInt64((fmi3Instance)in, vr, 1, i64, 1) == fmi3OK);
    });
    CHECK(i64[0] == INT64_MIN);
    fmi3Boolean b[1] = { fmi3False };
    const char *bad_bool[] = { "[0.5]", "[2]", "[-1]", "[\"NaN\"]" };
    for (size_t k = 0; k < sizeof bad_bool / sizeof *bad_bool; ++k) {
        char reply[96]; snprintf(reply, sizeof reply, "{\"ok\":true,\"values\":%s}", bad_bool[k]);
        WITH_SERVER(reply, 0, {
            CHECK(fmi3GetBoolean((fmi3Instance)in, vr, 1, b, 1) == fmi3Error);
        });
        CHECK(b[0] == fmi3False);
    }
    CHECK(strstr(g_last_log, "not a value of type Boolean") != NULL);
    /* Float32: a finite value beyond FLT_MAX is refused (converting it is
     * undefined); a non-finite one is a value the type holds */
    fmi3Float32 f32[1] = { 1.0f };
    WITH_SERVER("{\"ok\":true,\"values\":[1e300]}", 0, {
        CHECK(fmi3GetFloat32((fmi3Instance)in, vr, 1, f32, 1) == fmi3Error);
    });
    CHECK(f32[0] == 1.0f);
    WITH_SERVER("{\"ok\":true,\"values\":[\"NaN\"]}", 0, {
        CHECK(fmi3GetFloat32((fmi3Instance)in, vr, 1, f32, 1) == fmi3OK);
    });
    CHECK(isnan(f32[0]));
    /* Float64 takes anything a double is */
    fmi3Float64 f64[1] = { 0 };
    WITH_SERVER("{\"ok\":true,\"values\":[1e300]}", 0, {
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 1, f64, 1) == fmi3OK && f64[0] == 1e300);
    });
    /* binary replies go through the same check */
    unsigned char frame[128], le[8];
    double half = 0.5;
    f64_to_le(le, &half, 1);
    size_t n = bin_payload(frame, "{\"ok\":true,\"n\":1,\"dtype\":\"f64\"}", le, 8);
    i32[0] = 42;
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetInt32((fmi3Instance)in, vr, 1, i32, 1) == fmi3Error);
    });
    CHECK(i32[0] == 42);
}

static void test_int64_setters_refuse_values_a_double_cannot_carry(void) {
    /* Each refusal is the wrapper's own, made with a peer that would have
     * answered "ok" to the rounded value. */
    fmi3ValueReference vr[1] = { 7 };
    fmi3Int64 big[1] = { 9007199254740993LL };                  /* 2^53 + 1 */
    g_log_calls = 0;
    WITH_WILLING_SERVER({ CHECK(fmi3SetInt64((fmi3Instance)in, vr, 1, big, 1) == fmi3Error); });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "cannot be carried exactly") != NULL);
    fmi3Int64 low[1] = { -9007199254740993LL };                 /* -(2^53 + 1) */
    WITH_WILLING_SERVER({ CHECK(fmi3SetInt64((fmi3Instance)in, vr, 1, low, 1) == fmi3Error); });
    fmi3Int64 top[1] = { INT64_MAX };                           /* rounds to 2^63 */
    WITH_WILLING_SERVER({ CHECK(fmi3SetInt64((fmi3Instance)in, vr, 1, top, 1) == fmi3Error); });
    fmi3UInt64 ubig[1] = { 9007199254740993ULL };
    WITH_WILLING_SERVER({ CHECK(fmi3SetUInt64((fmi3Instance)in, vr, 1, ubig, 1) == fmi3Error); });
    fmi3UInt64 utop[1] = { UINT64_MAX };                        /* rounds to 2^64 */
    g_log_calls = 0;
    WITH_WILLING_SERVER({ CHECK(fmi3SetUInt64((fmi3Instance)in, vr, 1, utop, 1) == fmi3Error); });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "cannot be carried exactly") != NULL);
    /* a refused value anywhere in the array refuses the whole call */
    fmi3Int64 mixed[3] = { 1, 9007199254740993LL, 2 };
    g_log_calls = 0;
    WITH_WILLING_SERVER({ CHECK(fmi3SetInt64((fmi3Instance)in, vr, 1, mixed, 3) == fmi3Error); });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "values[1]") != NULL);
    /* exact ones reach the wire, as they are */
    fmi3Int64 ok[2] = { 9007199254740992LL, INT64_MIN };
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetInt64((fmi3Instance)in, vr, 1, ok, 2) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"set\",\"type\":\"Int64\",\"vr\":[7],"
                         "\"values\":[9007199254740992,-9.2233720368547758e+18]}") == 0);
}

/* ------------------------------------------ SetFMUState: the whole frame */

static void test_set_fmu_state_counts_the_frame_header(void) {
    /* The bridge drops a connection whose frame exceeds FRAME_MAX.  The
     * binary set_state frame is [u32 hl][header][blob]; the check used to
     * be on the blob alone, so a blob of exactly FRAME_MAX bytes passed it
     * and the instance died ("send failed", then every call closed).
     * Every case runs against a peer that takes what it is sent: a refusal
     * is the wrapper's, and the largest frame that fits arrives whole. */
    char hdr[64];
    size_t n = FRAME_MAX, hl;
    for (;;) {             /* the largest blob whose whole frame fits */
        hl = (size_t)snprintf(hdr, sizeof hdr, "{\"op\":\"set_state\",\"n\":%lu}", (unsigned long)n);
        if (4 + hl + n <= FRAME_MAX) break;
        --n;
    }
    char *blob = (char *)calloc(FRAME_MAX + 2, 1);   /* never read past what each case names */
    FmuState fits = bare_state(blob, n), over = bare_state(blob, n + 1),
             whole = bare_state(blob, FRAME_MAX);
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(fmi3SetFMUState((fmi3Instance)in, &whole) == fmi3Error && in->req_cap == 0);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        in->binary = 1;
        CHECK(fmi3SetFMUState((fmi3Instance)in, &over) == fmi3Error && in->req_cap == 0);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
    g_log_calls = 0;
    WITH_SERVER("{\"ok\":true}", 0, {
        in->binary = 1;
        CHECK(fmi3SetFMUState((fmi3Instance)in, &fits) == fmi3OK);     /* reaches the wire */
    });
    CHECK(g_log_calls == 0 && g_seen_flag == 1 && g_seen_len == FRAME_MAX);   /* exactly the limit */
    CHECK(get_be32((const unsigned char *)g_seen) == hl && memcmp(g_seen + 4, hdr, hl) == 0);
    /* the JSON path: {"op":"set_state","state":"<blob>"} */
    const size_t overhead = strlen("{\"op\":\"set_state\",\"state\":\"\"}");
    size_t jn = FRAME_MAX - overhead;
    memset(blob, 'A', jn + 1);
    FmuState jfits = bare_state(blob, jn), jover = bare_state(blob, jn + 1);
    g_log_calls = 0;
    WITH_WILLING_SERVER({
        CHECK(fmi3SetFMUState((fmi3Instance)in, &jover) == fmi3Error && in->req_cap == 0);
    });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
    g_log_calls = 0;
    blob[jn] = '\0';
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetFMUState((fmi3Instance)in, &jfits) == fmi3OK);
    });
    CHECK(g_log_calls == 0 && g_seen_flag == 0 && g_seen_len == FRAME_MAX);   /* exactly the limit, sent */
    free(blob);
}

/* ------------------------------------------------------------ deadlines */

static void test_read_timeout(void) {
    double t = -1;
    Instance *in = fake_instance(SOCK_INVALID);
    unsetenv("MADDENING_FMU_TIMEOUT");
    CHECK(read_timeout(in, &t) == 0 && t == REPLY_TIMEOUT_DEFAULT_S && t == 600.0);
    setenv("MADDENING_FMU_TIMEOUT", "", 1);
    CHECK(read_timeout(in, &t) == 0 && t == 600.0);
    setenv("MADDENING_FMU_TIMEOUT", "2.5", 1);       /* '.' whatever the locale */
    CHECK(read_timeout(in, &t) == 0 && t == 2.5);
    setenv("MADDENING_FMU_TIMEOUT", "0", 1);
    CHECK(read_timeout(in, &t) == 0 && t == 0.0);
    setenv("MADDENING_FMU_TIMEOUT", "1e6", 1);
    CHECK(read_timeout(in, &t) == 0 && t == 1e6);
    const char *bad[] = { "-1", "abc", "nan", "inf", "1e7", "5x", "0x", "2,5" };
    for (size_t k = 0; k < sizeof bad / sizeof *bad; ++k) {
        setenv("MADDENING_FMU_TIMEOUT", bad[k], 1);
        t = 123;
        CHECK(read_timeout(in, &t) == -1 && t == 123);
    }
    unsetenv("MADDENING_FMU_TIMEOUT");
    free_instance(in);
}

/* ------------------------------------------ numbers in the C locale */

static void test_numbers_are_written_and_read_in_the_c_locale(void) {
    /* Whatever LC_NUMERIC the importer runs under, the wire carries '.'
     * decimals: %.17g and strtod go through the instance's C locale.  This
     * runs under the process locale main() set from the environment;
     * tests/fmi/test_c_unit.py runs the whole binary under a ',' locale
     * too, where every check in this file is a check of it.  Here: a
     * doStep request, an initialize request, a set request and a reply. */
    fmi3Boolean ev, term, early; fmi3Float64 last;
    WITH_SERVER("{\"ok\":true,\"t\":0.51}", 0, {
        CHECK(fmi3DoStep((fmi3Instance)in, 0.5, 0.01, fmi3False, &ev, &term, &early, &last) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"step\",\"t\":0.5,\"dt\":0.01}") == 0);
    WITH_SERVER("{\"ok\":true,\"t\":0.25}", 0, {
        in->phase = PHASE_INSTANTIATED;
        CHECK(fmi3EnterInitializationMode((fmi3Instance)in, fmi3False, 0, 0.25, fmi3False, 0) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"initialize\",\"t\":0.25}") == 0);
    fmi3ValueReference vr[1] = { 7 };
    fmi3Float64 v[1] = { 2.5 };
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3SetFloat64((fmi3Instance)in, vr, 1, v, 1) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"set\",\"type\":\"Float64\",\"vr\":[7],\"values\":[2.5]}") == 0);
    fmi3Float64 got[1] = { 0 };
    WITH_SERVER("{\"ok\":true,\"values\":[0.125]}", 0, {
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 1, got, 1) == fmi3OK && got[0] == 0.125);
    });
    /* and the process locale is left as it was */
    printf("decimal point after the C-locale calls: %s\n", localeconv()->decimal_point);
}

/* ------------------------------------- the wrapper's clock (lastSuccessfulTime) */

static void test_the_wrappers_clock_follows_set_fmu_state_and_reset(void) {
    fmi3FMUState st = NULL;
    fmi3DeserializeFMUState(NULL, (const fmi3Byte *)"QUJDRA==", 8, &st);
    fmi3Boolean ev, term, early; fmi3Float64 last = -1;
    /* a restore moves the clock to the restored time, which a refused
     * doStep then reports as lastSuccessfulTime (it reported the time
     * before the restore) */
    WITH_SERVER("{\"ok\":true,\"t\":0.05}", 0, {
        in->time = 0.1;
        CHECK(fmi3SetFMUState((fmi3Instance)in, st) == fmi3OK);
        CHECK(in->time == 0.05);
    });
    WITH_SERVER("{\"ok\":false,\"error\":\"ValueError: not a whole multiple\"}", 0, {
        in->time = 0.05;
        CHECK(fmi3DoStep((fmi3Instance)in, 0.05, 0.015, fmi3False, &ev, &term, &early, &last) == fmi3Error);
        CHECK(last == 0.05);
    });
    /* an older bridge's reply has no "t": the clock is left alone */
    WITH_SERVER("{\"ok\":true}", 0, {
        in->time = 0.1;
        CHECK(fmi3SetFMUState((fmi3Instance)in, st) == fmi3OK && in->time == 0.1);
    });
    /* nor does a refused restore move it */
    WITH_SERVER("{\"ok\":false,\"error\":\"ValueError: token\",\"t\":9}", 0, {
        in->time = 0.1;
        CHECK(fmi3SetFMUState((fmi3Instance)in, st) == fmi3Error && in->time == 0.1);
    });
    /* the binary path: the reply is a JSON frame either way */
    {
        unsigned char blob[4] = { 'P', 'K', 3, 4 };
        fmi3FMUState raw = NULL;
        fmi3DeserializeFMUState(NULL, blob, 4, &raw);
        WITH_SERVER("{\"ok\":true,\"t\":0.07}", 0, {
            in->binary = 1; in->time = 0.0;
            CHECK(fmi3SetFMUState((fmi3Instance)in, raw) == fmi3OK && in->time == 0.07);
        });
        fmi3FreeFMUState(NULL, &raw);
    }
    fmi3FreeFMUState(NULL, &st);
    /* reset: the clock goes to zero only once the bridge has reset */
    WITH_SERVER("{\"ok\":false,\"error\":\"RuntimeError: stopped\"}", 0, {
        in->time = 0.3;
        CHECK(fmi3Reset((fmi3Instance)in) == fmi3Error && in->time == 0.3);
    });
    WITH_SERVER("{\"ok\":true}", 0, {
        in->time = 0.3;
        CHECK(fmi3Reset((fmi3Instance)in) == fmi3OK && in->time == 0.0);
    });
}

static void test_a_silent_sidecar_times_out(void) {
    /* A peer that takes the request and never answers: the wrapper used to
     * wait in recv for ever.  With a deadline the call is fmi3Error, the
     * connection is closed (a reply arriving later would be out of step),
     * and the log says why. */
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    Instance *in = fake_instance(sv[0]);
    CHECK(set_socket_timeout(sv[0], 0.2) == 0);
    g_log_calls = 0;
    CHECK(bridge_call(in, "{\"op\":\"step\"}") == fmi3Error);
    CHECK(in->sock == SOCK_INVALID);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "did not answer within the deadline") != NULL);
    /* a peer that hangs up is still reported as a hang-up, not a timeout */
    int sv2[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv2) == 0);
    Instance *gone = fake_instance(sv2[0]);
    CHECK(set_socket_timeout(sv2[0], 5.0) == 0);
    shutdown(sv2[1], SHUT_WR);
    g_log_calls = 0;
    CHECK(bridge_call(gone, "{\"op\":\"step\"}") == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "recv failed") != NULL);
    sock_close(sv[1]); sock_close(sv2[1]);
    free_instance(in); free_instance(gone);
    /* 0 removes the deadline */
    int sv3[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv3) == 0);
    CHECK(set_socket_timeout(sv3[0], 0.0) == 0);
    struct timeval tv; socklen_t len = sizeof tv;
    CHECK(getsockopt(sv3[0], SOL_SOCKET, SO_RCVTIMEO, &tv, &len) == 0 && tv.tv_sec == 0 && tv.tv_usec == 0);
    sock_close(sv3[0]); sock_close(sv3[1]);
}

/* ----------------------------------------------------------- FMU state */

static void test_fmu_state(void) {
    Instance *in = fake_instance(SOCK_INVALID);     /* the states below are its */
    fmi3FMUState st = NULL;
    EXCHANGE(in, "{\"ok\":true,\"state\":\"QUJDRA==\"}", {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK);
    });
    CHECK(st != NULL && ((FmuState *)st)->n == 8 && memcmp(((FmuState *)st)->blob, "QUJDRA==", 8) == 0);
    size_t n = 0;
    CHECK(fmi3SerializedFMUStateSize((fmi3Instance)in, st, &n) == fmi3OK && n == 8);
    fmi3Byte buf[8];
    CHECK(fmi3SerializeFMUState((fmi3Instance)in, st, buf, 4) == fmi3Error);      /* too small */
    CHECK(fmi3SerializeFMUState((fmi3Instance)in, st, buf, 8) == fmi3OK);
    fmi3FMUState st2 = NULL;
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, buf, 8, &st2) == fmi3OK);
    CHECK(memcmp(((FmuState *)st2)->blob, "QUJDRA==", 8) == 0);
    EXCHANGE(in, "{\"ok\":true}", {
        CHECK(fmi3SetFMUState((fmi3Instance)in, st2) == fmi3OK);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"set_state\",\"state\":\"QUJDRA==\"}") == 0);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &st) == fmi3OK && st == NULL);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &st2) == fmi3OK);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &st) == fmi3OK);        /* a freed (NULL) one is ignored */
    /* missing / unterminated state field */
    EXCHANGE(in, "{\"ok\":true}", {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3Error);
    });
    EXCHANGE(in, "{\"ok\":true,\"state\":\"unterminated", {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3Error);
    });
    CHECK(st == NULL);
    WITH_WILLING_SERVER({ CHECK(fmi3SetFMUState((fmi3Instance)in, NULL) == fmi3Error); });
    CHECK(fmi3SerializedFMUStateSize((fmi3Instance)in, NULL, &n) == fmi3Error);
    /* importer-supplied bytes that are not base64 never reach the wire */
    fmi3FMUState junk = NULL;
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, (const fmi3Byte *)"abc\"def\\x", 9, &junk) == fmi3OK);
    g_log_calls = 0;
    WITH_WILLING_SERVER({ CHECK(fmi3SetFMUState((fmi3Instance)in, junk) == fmi3Error); });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "not valid") != NULL);
    fmi3FreeFMUState((fmi3Instance)in, &junk);
    free_instance(in);
}

/* ------------------------------------- FMU states: who owns the memory
 *
 * FMI 3.0.1, "Getting and Setting the Complete FMU State", function by
 * function (the rules are quoted at the top of the wrapper).  The oracle
 * is the count of live allocations: a state is two (the object and its
 * blob), and nothing else here allocates once the instance's reply buffer
 * exists.
 */

#define STATE_ALLOCS 2L

/* The instance's live states, head first, as a count; -1 if the list's
 * links or owners are inconsistent. */
static long states_of(const Instance *in) {
    long k = 0;
    const FmuState *prev = NULL;
    for (const FmuState *st = in->states; st; prev = st, st = st->next, ++k)
        if (st->owner != in || st->prev != prev) return -1;
    return k;
}

static void test_get_fmu_state_reuses_the_state_object_it_is_handed(void) {
    /* The rollback pattern: one fmi3FMUState variable, fmi3GetFMUState(&state)
     * before every step.  FMI 3.0: a non-NULL *FMUState "points to a
     * previously returned FMUState that is no longer needed and can be
     * overwritten".  The wrapper allocated a new object and blob over it on
     * every call -- one whole state leaked per call (2.8 kB a call for one
     * spring, 82 kB for a 20000-cell rod) -- and returned another pointer. */
    Instance *in = fake_instance(SOCK_INVALID);
    fmi3FMUState state = NULL;
    EXCHANGE(in, "{\"ok\":true,\"state\":\"QUJDRA==\"}", {
        CHECK(fmi3GetFMUState((fmi3Instance)in, &state) == fmi3OK);
    });
    CHECK(state != NULL && states_of(in) == 1);
    const void *first = state;
    const long held = live_allocs();                 /* the state and the reply buffer */
    const char *replies[] = { "{\"ok\":true,\"state\":\"QUJDREVGR0hJSktMTU5PUA==\"}",   /* longer */
                              "{\"ok\":true,\"state\":\"QQ==\"}",                       /* shorter */
                              "{\"ok\":true,\"state\":\"\"}",                           /* empty */
                              "{\"ok\":true,\"state\":\"QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=\"}" };
    for (int k = 0; k < 40; ++k) {
        const char *reply = replies[k % 4];
        EXCHANGE(in, reply, { CHECK(fmi3GetFMUState((fmi3Instance)in, &state) == fmi3OK); });
        CHECK(state == first);                       /* the same object, as FMI describes */
        CHECK(live_allocs() == held);                /* and nothing new left behind */
        const char *want = strstr(reply, "\"state\":\"") + 9;
        size_t n = (size_t)(strchr(want, '"') - want);
        FmuState *st = (FmuState *)state;
        CHECK(st->n == n && memcmp(st->blob, want, n) == 0 && st->blob[n] == '\0');
        CHECK(states_of(in) == 1);
    }
    /* the binary path reuses it too */
    {
        unsigned char blob[6] = { 'P', 'K', 0, 7, 0, 9 }, frame[64];
        size_t n = bin_payload(frame, "{\"ok\":true,\"n\":6}", blob, 6);
        in->binary = 1;
        BINARY_EXCHANGE(in, (const char *)frame, n, {
            CHECK(fmi3GetFMUState((fmi3Instance)in, &state) == fmi3OK);
        });
        in->binary = 0;
        CHECK(state == first && live_allocs() == held);
        CHECK(((FmuState *)state)->n == 6 && memcmp(((FmuState *)state)->blob, blob, 6) == 0);
    }
    /* A call that fails leaves the variable, and the state it holds, as
     * they were: a refusal, a reply without a state, a length mismatch. */
    {
        unsigned char frame[64];
        size_t n = bin_payload(frame, "{\"ok\":true,\"n\":5}", "PK\0\7\0\11", 6);
        const char *bad[] = { "{\"ok\":false,\"error\":\"RuntimeError: stopped\"}", "{\"ok\":true}",
                              "{\"ok\":true,\"state\":\"unterminated" };
        for (size_t k = 0; k < sizeof bad / sizeof *bad; ++k) {
            EXCHANGE(in, bad[k], { CHECK(fmi3GetFMUState((fmi3Instance)in, &state) == fmi3Error); });
            CHECK(state == first && live_allocs() == held);
            CHECK(((FmuState *)state)->n == 6 && memcmp(((FmuState *)state)->blob, "PK\0\7\0\11", 6) == 0);
        }
        BINARY_EXCHANGE(in, (const char *)frame, n, {
            CHECK(fmi3GetFMUState((fmi3Instance)in, &state) == fmi3Error);
        });
        CHECK(state == first && live_allocs() == held && ((FmuState *)state)->n == 6);
    }
    /* Out of memory while a held state must grow: fmi3Fatal, and the state
     * is still the one the importer had.  (The reply buffer is already
     * large enough, so the allocation that fails is the state's.) */
    {
        char big[600];
        memset(big, 'Q', sizeof big);
        memcpy(big, "{\"ok\":true,\"pad\":\"", 18);
        memcpy(big + sizeof big - 3, "\"}", 3);
        EXCHANGE(in, big, { CHECK(bridge_call(in, "{\"op\":\"x\"}") == fmi3OK); });
        const long with_buffer = live_allocs();
        memcpy(big, "{\"ok\":true,\"state\":\"", 20);
        EXCHANGE(in, big, {
            g_fail_allocs_in = 0;
            CHECK(fmi3GetFMUState((fmi3Instance)in, &state) == fmi3Fatal);
            CHECK(g_fail_allocs_in < 0);             /* the failure was taken */
        });
        g_fail_allocs_in = -1;
        CHECK(state == first && live_allocs() == with_buffer);
        CHECK(((FmuState *)state)->n == 6 && memcmp(((FmuState *)state)->blob, "PK\0\7\0\11", 6) == 0);
        /* and with no state yet: neither allocation that can fail leaves
         * anything behind, and the variable stays NULL */
        for (long nth = 0; nth < 2; ++nth) {
            fmi3FMUState fresh = NULL;
            EXCHANGE(in, big, {
                g_fail_allocs_in = nth;
                CHECK(fmi3GetFMUState((fmi3Instance)in, &fresh) == fmi3Fatal);
                CHECK(g_fail_allocs_in < 0);
            });
            g_fail_allocs_in = -1;
            CHECK(fresh == NULL && live_allocs() == with_buffer && states_of(in) == 1);
        }
    }
    /* A pointer that is not a live state of this instance is not taken for
     * one: it is overwritten unread (as every pointer used to be), so an
     * uninitialised variable does not become a wild free. */
    {
        const long before = live_allocs();
        fmi3FMUState wild = (fmi3FMUState)(uintptr_t)0x10;      /* not a pointer to anything */
        EXCHANGE(in, "{\"ok\":true,\"state\":\"QQ==\"}", {
            CHECK(fmi3GetFMUState((fmi3Instance)in, &wild) == fmi3OK);
        });
        CHECK(wild != (fmi3FMUState)(uintptr_t)0x10 && wild != state && states_of(in) == 2);
        CHECK(live_allocs() == before + STATE_ALLOCS);
        CHECK(fmi3FreeFMUState((fmi3Instance)in, &wild) == fmi3OK && live_allocs() == before);
    }
    /* fmi3GetFMUState(instance, NULL) has nowhere to return a state */
    g_log_calls = 0;
    WITH_WILLING_SERVER({ CHECK(fmi3GetFMUState((fmi3Instance)in, NULL) == fmi3Error); });
    CHECK(g_log_calls == 1 && strstr(g_last_log, "FMUState is NULL") != NULL);
    const long before_free = live_allocs();
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &state) == fmi3OK && state == NULL);
    CHECK(live_allocs() == before_free - STATE_ALLOCS && states_of(in) == 0);
    free_instance(in);
}

static void test_the_fmu_state_functions_keep_to_fmi_ownership(void) {
    Instance *in = fake_instance(SOCK_INVALID);
    Instance *other = fake_instance(SOCK_INVALID);
    const long base = live_allocs();
    fmi3FMUState a = NULL;
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, (const fmi3Byte *)"QUJD", 4, &a) == fmi3OK);
    CHECK(live_allocs() == base + STATE_ALLOCS && states_of(in) == 1);

    /* fmi3SetFMUState: "The FMU must not change the content of the provided
     * FMUState to allow multiple calls" -- the object is the same, bit for
     * bit, after two restores and after a refused one. */
    FmuState seen = *(FmuState *)a;
    for (int k = 0; k < 2; ++k) {
        EXCHANGE(in, "{\"ok\":true,\"t\":0.5}", { CHECK(fmi3SetFMUState((fmi3Instance)in, a) == fmi3OK); });
        CHECK(strcmp(g_seen, "{\"op\":\"set_state\",\"state\":\"QUJD\"}") == 0);
        CHECK(memcmp(a, &seen, sizeof seen) == 0 && memcmp(((FmuState *)a)->blob, "QUJD", 5) == 0);
    }
    EXCHANGE(in, "{\"ok\":false,\"error\":\"ValueError: token\"}", {
        CHECK(fmi3SetFMUState((fmi3Instance)in, a) == fmi3Error);
    });
    CHECK(memcmp(a, &seen, sizeof seen) == 0 && live_allocs() == base + STATE_ALLOCS + 2);  /* + req, resp */
    const long held = live_allocs();

    /* fmi3SerializedFMUStateSize / fmi3SerializeFMUState: the object is read,
     * and only `n` bytes of the importer's vector are written. */
    size_t n = 99;
    CHECK(fmi3SerializedFMUStateSize((fmi3Instance)in, a, &n) == fmi3OK && n == 4);
    CHECK(fmi3SerializedFMUStateSize((fmi3Instance)in, a, NULL) == fmi3Error);
    fmi3Byte vec[8];
    memset(vec, 0xAA, sizeof vec);
    CHECK(fmi3SerializeFMUState((fmi3Instance)in, a, vec, 3) == fmi3Error && vec[0] == 0xAA);
    CHECK(fmi3SerializeFMUState((fmi3Instance)in, a, NULL, 4) == fmi3Error);
    CHECK(fmi3SerializeFMUState((fmi3Instance)in, NULL, vec, 8) == fmi3Error && vec[0] == 0xAA);
    CHECK(fmi3SerializeFMUState((fmi3Instance)in, a, vec, 8) == fmi3OK);
    CHECK(memcmp(vec, "QUJD", 4) == 0 && vec[4] == 0xAA && vec[7] == 0xAA);
    CHECK(memcmp(a, &seen, sizeof seen) == 0 && live_allocs() == held);

    /* fmi3DeserializeFMUState "constructs a copy": always a new object.
     * The standard says nothing about *FMUState on entry, so it is not
     * read -- a variable still holding a live state gets a second state
     * (the first stays valid, and the importer's to free), and so does one
     * holding no pointer at all. */
    fmi3FMUState b = a;
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, (const fmi3Byte *)"WFla", 4, &b) == fmi3OK);
    CHECK(b != a && memcmp(a, &seen, offsetof(FmuState, prev)) == 0 && states_of(in) == 2);
    CHECK(memcmp(((FmuState *)a)->blob, "QUJD", 4) == 0 && memcmp(((FmuState *)b)->blob, "WFla", 4) == 0);
    fmi3FMUState c = (fmi3FMUState)(uintptr_t)0x10;
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, (const fmi3Byte *)"", 0, &c) == fmi3OK);
    CHECK(c != (fmi3FMUState)(uintptr_t)0x10 && ((FmuState *)c)->n == 0 && states_of(in) == 3);
    CHECK(live_allocs() == held + 2 * STATE_ALLOCS);
    /* an empty vector may be NULL; a NULL one that claims bytes, or nowhere
     * to return the state, is refused with nothing allocated */
    fmi3FMUState d = NULL;
    g_log_calls = 0;
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, NULL, 4, &d) == fmi3Error && d == NULL);
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, (const fmi3Byte *)"QUJD", 4, NULL) == fmi3Error);
    CHECK(g_log_calls == 2 && live_allocs() == held + 2 * STATE_ALLOCS);
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, NULL, 0, &d) == fmi3OK && ((FmuState *)d)->n == 0);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &d) == fmi3OK && d == NULL);
    /* a vector no frame could carry is refused before it is copied (size + 1
     * bytes used to be allocated, whatever the size, and filled from it) */
    g_log_calls = 0;
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, (const fmi3Byte *)"x", (size_t)FRAME_MAX + 1, &d) == fmi3Error);
    CHECK(fmi3DeserializeFMUState((fmi3Instance)in, (const fmi3Byte *)"x", SIZE_MAX, &d) == fmi3Error);
    CHECK(d == NULL && g_log_calls == 2 && strstr(g_last_log, "frame limit") != NULL);
    CHECK(live_allocs() == held + 2 * STATE_ALLOCS && states_of(in) == 3);

    /* fmi3GetFMUState on another instance does not take this instance's
     * state for one of its own: `a` stays what it was, and whose it was. */
    fmi3FMUState borrowed = a;
    EXCHANGE(other, "{\"ok\":true,\"state\":\"Wlpa\"}", {
        CHECK(fmi3GetFMUState((fmi3Instance)other, &borrowed) == fmi3OK);
    });
    CHECK(borrowed != a && memcmp(((FmuState *)a)->blob, "QUJD", 4) == 0);
    CHECK(states_of(in) == 3 && states_of(other) == 1 && ((FmuState *)a)->owner == in);

    /* fmi3Reset and fmi3Terminate leave every state object valid: the
     * state saved before restores after. */
    EXCHANGE(in, "{\"ok\":true}", { CHECK(fmi3Terminate((fmi3Instance)in) == fmi3OK); });
    EXCHANGE(in, "{\"ok\":true}", { CHECK(fmi3Reset((fmi3Instance)in) == fmi3OK); });
    CHECK(states_of(in) == 3 && memcmp(((FmuState *)a)->blob, "QUJD", 5) == 0);
    EXCHANGE(in, "{\"ok\":true,\"t\":0.5}", { CHECK(fmi3SetFMUState((fmi3Instance)in, a) == fmi3OK); });
    CHECK(strcmp(g_seen, "{\"op\":\"set_state\",\"state\":\"QUJD\"}") == 0);

    /* fmi3FreeFMUState: the middle, the head and the tail of the list, a
     * second free of the (now NULL) variable, and a NULL argument. */
    const long before_free = live_allocs();
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &b) == fmi3OK && b == NULL && states_of(in) == 2);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &b) == fmi3OK && b == NULL && states_of(in) == 2);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, NULL) == fmi3OK && states_of(in) == 2);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &c) == fmi3OK && c == NULL && states_of(in) == 1);
    CHECK(in->states == (FmuState *)a);
    CHECK(fmi3FreeFMUState(NULL, &a) == fmi3OK && a == NULL && states_of(in) == 0);   /* any instance */
    CHECK(live_allocs() == before_free - 3 * STATE_ALLOCS);
    CHECK(fmi3FreeFMUState((fmi3Instance)other, &borrowed) == fmi3OK && states_of(other) == 0);
    free_instance(in);
    free_instance(other);
}

static void test_free_instance_frees_the_states_that_are_still_live(void) {
    /* FMI 3.0: fmi3FreeInstance "frees all the allocated memory and other
     * resources that have been allocated by the functions of the FMU
     * interface".  A state the importer never freed used to outlive its
     * instance -- and the library, once the importer unloaded it. */
    const long base = live_allocs();
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    ServerArgs args = { sv[1], "{\"ok\":true}", 11, 11, 0, NULL, 0, 0, 0 };
    pthread_t th; pthread_create(&th, NULL, server_thread, &args);
    Instance *in = fake_instance(sv[0]);
    fmi3FMUState kept[5] = { NULL, NULL, NULL, NULL, NULL };
    for (int k = 0; k < 5; ++k)
        CHECK(fmi3DeserializeFMUState((fmi3Instance)in, (const fmi3Byte *)"QUJDRA==", 8, &kept[k]) == fmi3OK);
    CHECK(states_of(in) == 5 && live_allocs() == base + 1 + 5 * STATE_ALLOCS);
    CHECK(fmi3FreeFMUState((fmi3Instance)in, &kept[2]) == fmi3OK);    /* one, by the importer */
    CHECK(states_of(in) == 4);
    fmi3FreeInstance((fmi3Instance)in);              /* tells the bridge, frees the rest */
    pthread_join(th, NULL);
    CHECK(args.seen != NULL && strcmp(args.seen, "{\"op\":\"terminate\"}") == 0);
    free(args.seen);
    sock_close(sv[1]);
    CHECK(live_allocs() == base);
    /* a state in no instance's list (made with a NULL instance) is the
     * importer's alone to free */
    fmi3FMUState loose = NULL;
    CHECK(fmi3DeserializeFMUState(NULL, (const fmi3Byte *)"QUJD", 4, &loose) == fmi3OK);
    CHECK(((FmuState *)loose)->owner == NULL && live_allocs() == base + STATE_ALLOCS);
    CHECK(fmi3FreeFMUState(NULL, &loose) == fmi3OK && live_allocs() == base);
}

/* ------------------------------------------------ instantiate via TCP */

typedef struct { int listen_fd; const char *hello_reply; int accepted; char *hello_seen;
                 char *second_seen; } Listener;
static void *listener_thread(void *p) {
    Listener *L = (Listener *)p;
    int c = accept(L->listen_fd, NULL, NULL);
    if (c < 0) return NULL;
    L->accepted = 1;
    char *req = fake_exchange(c, L->hello_reply, 0);      /* hello */
    free(L->hello_seen); L->hello_seen = req;
    req = fake_exchange(c, "{\"ok\":true}", 0);          /* initialize, or terminate */
    free(L->second_seen); L->second_seen = req;
    sock_close(c);
    return NULL;
}

/* Accepts one connection, reads the hello and never answers; records
 * whether the client then hung up. */
typedef struct { int listen_fd; int accepted; int saw_eof; } Silent;
static void *silent_thread(void *p) {
    Silent *S = (Silent *)p;
    int c = accept(S->listen_fd, NULL, NULL);
    if (c < 0) return NULL;
    S->accepted = 1;
    char buf[256];
    for (;;) {
        ssize_t k = recv(c, buf, sizeof buf, 0);
        if (k <= 0) { S->saw_eof = (k == 0); break; }
    }
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

    /* an old (protocol-1) bridge: its hello lacks "protocol" -> JSON everywhere */
    char reply[256]; snprintf(reply, sizeof reply, "{\"ok\":true,\"token\":\"%s\",\"model\":\"m\"}", good_token);
    Listener L = { fd, reply, 0, NULL, NULL };
    pthread_t th; pthread_create(&th, NULL, listener_thread, &L);
    fmi3Instance inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3True,
                                                    fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst != NULL);
    if (inst) {
        Instance *in = (Instance *)inst;
        CHECK(strcmp(in->instance_name, "i") == 0 && in->logging_on == fmi3True);
        CHECK(in->binary == 0);
        /* Nagle is off: every call is a small request awaiting its reply,
         * sent as two writes, and with Nagle the second waited for the
         * peer's delayed ACK (>= 40 ms per FMI call on Linux) */
        {
            int nodelay = 0; socklen_t len = sizeof nodelay;
            CHECK(getsockopt(in->sock, IPPROTO_TCP, TCP_NODELAY, &nodelay, &len) == 0 && nodelay != 0);
        }
        /* the full reply deadline (default 600 s) once the hello is done */
        {
            struct timeval tv; socklen_t len = sizeof tv;
            CHECK(in->timeout_s == 600.0);
            CHECK(getsockopt(in->sock, SOL_SOCKET, SO_RCVTIMEO, &tv, &len) == 0 && tv.tv_sec == 600);
        }
        CHECK(in->phase == PHASE_INSTANTIATED);         /* where FMI starts an instance */
        CHECK(fmi3EnterInitializationMode(inst, fmi3False, 0, 0.5, fmi3False, 0) == fmi3OK && in->time == 0.5);
        CHECK(in->phase == PHASE_INITIALIZATION_MODE);
        CHECK(fmi3ExitInitializationMode(inst) == fmi3OK && in->phase == PHASE_STEP_MODE);
        /* the listener has hung up after two exchanges, so FreeInstance's
         * terminate finds a closed peer: it must still free cleanly */
        fmi3FreeInstance(inst);
    }
    pthread_join(th, NULL);
    CHECK(L.accepted == 1);
    /* the client always offers protocol 2 */
    CHECK(L.hello_seen != NULL && strcmp(L.hello_seen, "{\"op\":\"hello\",\"protocol\":2,\"binary\":true}") == 0);
    /* and tells the bridge the start time */
    CHECK(L.second_seen != NULL && strcmp(L.second_seen, "{\"op\":\"initialize\",\"t\":0.5}") == 0);
    free(L.hello_seen); free(L.second_seen);

    /* a protocol-2 bridge that confirms binary frames */
    char reply2[256];
    snprintf(reply2, sizeof reply2,
             "{\"ok\":true,\"token\":\"%s\",\"model\":\"m\",\"master_dt\":0.01,\"protocol\":2,\"binary\":true}",
             good_token);
    Listener Lb = { fd, reply2, 0, NULL, NULL };
    pthread_create(&th, NULL, listener_thread, &Lb);
    inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3False,
                                       fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst != NULL);
    if (inst) { CHECK(((Instance *)inst)->binary == 1); fmi3FreeInstance(inst); }
    pthread_join(th, NULL);
    free(Lb.hello_seen); free(Lb.second_seen);

    /* protocol 2 announced but binary declined: JSON */
    snprintf(reply2, sizeof reply2,
             "{\"ok\":true,\"token\":\"%s\",\"protocol\":2,\"binary\":false}", good_token);
    Listener Lc = { fd, reply2, 0, NULL, NULL };
    pthread_create(&th, NULL, listener_thread, &Lc);
    inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3False,
                                       fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst != NULL);
    if (inst) { CHECK(((Instance *)inst)->binary == 0); fmi3FreeInstance(inst); }
    pthread_join(th, NULL);
    free(Lc.hello_seen); free(Lc.second_seen);

    /* a bridge that refuses the protocol -> NULL */
    Listener Ld = { fd, "{\"ok\":false,\"error\":\"protocol 2 is not supported\"}", 0, NULL, NULL };
    pthread_create(&th, NULL, listener_thread, &Ld);
    g_log_calls = 0;
    inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3False,
                                       fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst == NULL && g_log_calls >= 1 && strstr(g_last_log, "not supported") != NULL);
    pthread_join(th, NULL);            /* its second exchange sees the client's EOF */
    free(Ld.hello_seen); free(Ld.second_seen);

    /* token mismatch -> NULL, with a log line; so is an empty or NULL
     * token (FMI requires the modelDescription's; an empty one used to
     * skip the check), and a token that is a prefix of the bridge's, or
     * has the bridge's as a prefix: the comparison is of whole strings */
    char longer[128]; snprintf(longer, sizeof longer, "%sX", good_token);
    char shorter[128]; snprintf(shorter, sizeof shorter, "%.*s", (int)strlen(good_token) - 1, good_token);
    const char *bad_tokens[] = { "wrong-token", "", NULL, longer, shorter };
    for (size_t k = 0; k < sizeof bad_tokens / sizeof *bad_tokens; ++k) {
        Listener L2 = { fd, reply, 0, NULL, NULL };
        pthread_create(&th, NULL, listener_thread, &L2);
        g_log_calls = 0;
        inst = fmi3InstantiateCoSimulation("i", bad_tokens[k], NULL, fmi3False, fmi3False, fmi3False,
                                           fmi3False, NULL, 0, NULL, test_logger, NULL);
        CHECK(inst == NULL && g_log_calls >= 1 && strstr(g_last_log, "token") != NULL);
        if (inst) fmi3FreeInstance(inst);
        pthread_join(th, NULL);            /* its second exchange sees the client's EOF */
        CHECK(L2.accepted == 1 && L2.second_seen == NULL);
        free(L2.hello_seen); free(L2.second_seen);
    }
    /* an endpoint that accepts and never answers: instantiation used to
     * block for ever; now the hello gets min(MADDENING_FMU_TIMEOUT, 30 s) */
    {
        Silent S = { fd, 0, -1 };
        pthread_create(&th, NULL, silent_thread, &S);
        setenv("MADDENING_FMU_TIMEOUT", "0.3", 1);
        g_log_calls = 0;
        inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3False, fmi3False,
                                           fmi3False, NULL, 0, NULL, test_logger, NULL);
        CHECK(inst == NULL && strstr(g_last_log, "did not answer within the deadline") != NULL);
        pthread_join(th, NULL);
        CHECK(S.accepted == 1 && S.saw_eof == 1);     /* the wrapper hung up */
        /* a deadline that is not a number fails instantiation, loudly */
        setenv("MADDENING_FMU_TIMEOUT", "soon", 1);
        g_log_calls = 0;
        inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3False, fmi3False,
                                           fmi3False, NULL, 0, NULL, test_logger, NULL);
        CHECK(inst == NULL && g_log_calls == 1 && strstr(g_last_log, "MADDENING_FMU_TIMEOUT") != NULL);
        unsetenv("MADDENING_FMU_TIMEOUT");
    }
    shutdown(fd, SHUT_RDWR); sock_close(fd);

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

/* ------------------------------------------------- the FMI 3.0 state machine */

/* `call` on an instance in `phase` is refused by the state machine itself:
 * fmi3Error with exactly one log message naming the state, nothing sent
 * and the state unchanged -- against a peer that would have answered "ok"
 * (on a dead socket every call fails, so the status could not tell). */
#define REFUSED_IN(phase_, call) do {                                            \
    g_log_calls = 0; g_last_log[0] = '\0';                                       \
    WITH_WILLING_SERVER({                                                        \
        in->phase = (phase_);                                                    \
        CHECK((call) == fmi3Error);                                              \
        CHECK(in->phase == (phase_));                                            \
    });                                                                          \
    CHECK(g_log_calls == 1 && strstr(g_last_log, "is not allowed in the") != NULL); \
} while (0)

static void test_the_fmi_state_machine(void) {
    fmi3ValueReference vr[1] = { 7 };
    fmi3Float64 v[1] = { 0.5 };
    fmi3Boolean ev, term, early; fmi3Float64 last;
#define DOSTEP(in) fmi3DoStep((fmi3Instance)(in), 0.0, 0.01, fmi3False, &ev, &term, &early, &last)
#define INIT(in) fmi3EnterInitializationMode((fmi3Instance)(in), fmi3False, 0, 0.0, fmi3False, 0)
    /* Instantiated: no step before initialization (it used to advance the
     * model), no reads, no exit from a mode it is not in, no terminate */
    REFUSED_IN(PHASE_INSTANTIATED, DOSTEP(in));
    CHECK(strstr(g_last_log, "fmi3DoStep is not allowed in the Instantiated state") != NULL);
    REFUSED_IN(PHASE_INSTANTIATED, fmi3GetFloat64((fmi3Instance)in, vr, 1, v, 1));
    REFUSED_IN(PHASE_INSTANTIATED, fmi3GetFloat64((fmi3Instance)in, vr, 0, v, 0));
    REFUSED_IN(PHASE_INSTANTIATED, fmi3ExitInitializationMode((fmi3Instance)in));
    REFUSED_IN(PHASE_INSTANTIATED, fmi3Terminate((fmi3Instance)in));
    REFUSED_IN(PHASE_INSTANTIATED, fmi3EnterStepMode((fmi3Instance)in));
    WITH_SERVER("{\"ok\":true}", 0, {                      /* a set is allowed */
        in->phase = PHASE_INSTANTIATED;
        CHECK(fmi3SetFloat64((fmi3Instance)in, vr, 1, v, 1) == fmi3OK);
    });
    /* Initialization Mode */
    REFUSED_IN(PHASE_INITIALIZATION_MODE, INIT(in));
    REFUSED_IN(PHASE_INITIALIZATION_MODE, DOSTEP(in));
    REFUSED_IN(PHASE_INITIALIZATION_MODE, fmi3Terminate((fmi3Instance)in));
    REFUSED_IN(PHASE_INITIALIZATION_MODE, fmi3EnterStepMode((fmi3Instance)in));
    WITH_SERVER("{\"ok\":true,\"values\":[2]}", 0, {      /* reads and writes are */
        in->phase = PHASE_INITIALIZATION_MODE;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 1, v, 1) == fmi3OK && v[0] == 2.0);
    });
    {
        Instance *in = fake_instance(SOCK_INVALID);
        in->phase = PHASE_INITIALIZATION_MODE;
        CHECK(fmi3ExitInitializationMode((fmi3Instance)in) == fmi3OK && in->phase == PHASE_STEP_MODE);
        free_instance(in);
    }
    /* Step Mode: fmi3EnterStepMode is Event Mode's way out, never this FMU's */
    REFUSED_IN(PHASE_STEP_MODE, INIT(in));
    REFUSED_IN(PHASE_STEP_MODE, fmi3ExitInitializationMode((fmi3Instance)in));
    REFUSED_IN(PHASE_STEP_MODE, fmi3EnterStepMode((fmi3Instance)in));
    CHECK(strstr(g_last_log, "Event Mode") != NULL);
    WITH_SERVER("{\"ok\":false,\"error\":\"RuntimeError: stopped\"}", 0, {
        CHECK(fmi3Terminate((fmi3Instance)in) == fmi3Error && in->phase == PHASE_STEP_MODE);
    });
    WITH_SERVER("{\"ok\":true}", 0, {
        CHECK(fmi3Terminate((fmi3Instance)in) == fmi3OK && in->phase == PHASE_TERMINATED);
    });
    CHECK(strcmp(g_seen, "{\"op\":\"terminate\"}") == 0);
    /* Terminated: reading and the FMU-state functions only, until reset.
     * A zero-length set, fmi3ExitInitializationMode and fmi3EnterStepMode
     * all answered fmi3OK here. */
    REFUSED_IN(PHASE_TERMINATED, fmi3SetFloat64((fmi3Instance)in, vr, 0, v, 0));
    CHECK(strstr(g_last_log, "fmi3SetFloat64 is not allowed in the Terminated state") != NULL);
    REFUSED_IN(PHASE_TERMINATED, fmi3SetFloat64((fmi3Instance)in, vr, 1, v, 1));
    REFUSED_IN(PHASE_TERMINATED, DOSTEP(in));
    REFUSED_IN(PHASE_TERMINATED, INIT(in));
    REFUSED_IN(PHASE_TERMINATED, fmi3ExitInitializationMode((fmi3Instance)in));
    REFUSED_IN(PHASE_TERMINATED, fmi3EnterStepMode((fmi3Instance)in));
    REFUSED_IN(PHASE_TERMINATED, fmi3Terminate((fmi3Instance)in));
    {
        Instance *in = fake_instance(SOCK_INVALID);
        in->phase = PHASE_TERMINATED;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 0, v, 0) == fmi3OK);
        free_instance(in);
    }
    WITH_SERVER("{\"ok\":true,\"state\":\"QUJD\"}", 0, {
        in->phase = PHASE_TERMINATED;
        fmi3FMUState st = NULL;
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK && in->phase == PHASE_TERMINATED);
        fmi3FreeFMUState(NULL, &st);
    });
    WITH_SERVER("{\"ok\":false,\"error\":\"RuntimeError: stopped\"}", 0, {
        in->phase = PHASE_TERMINATED;
        CHECK(fmi3Reset((fmi3Instance)in) == fmi3Error && in->phase == PHASE_TERMINATED);
    });
    WITH_SERVER("{\"ok\":true}", 0, {
        in->phase = PHASE_TERMINATED;
        CHECK(fmi3Reset((fmi3Instance)in) == fmi3OK && in->phase == PHASE_INSTANTIATED);
    });
    /* a state the wrapper does not know allows nothing */
    REFUSED_IN(7, fmi3GetFloat64((fmi3Instance)in, vr, 1, v, 1));
    CHECK(strstr(g_last_log, "unknown state") != NULL);
    REFUSED_IN(-1, fmi3SetFloat64((fmi3Instance)in, vr, 1, v, 1));
#undef DOSTEP
#undef INIT
}

/* ------------------------------------------------- misc entry points */

static void test_misc_entry_points(void) {
    CHECK(strcmp(fmi3GetVersion(), "3.0") == 0);
    size_t n = 99;
    CHECK(fmi3GetNumberOfEventIndicators(NULL, &n) == fmi3OK && n == 0);
    CHECK(fmi3GetNumberOfContinuousStates(NULL, &n) == fmi3OK && n == 0);
    fmi3ValueReference vr[2] = { 1, 2 };
    fmi3Clock clk[2] = { fmi3ClockActive, fmi3ClockActive };
    /* no Event Mode, so no fmi3GetClock / fmi3SetClock (they answered
     * fmi3OK for any value reference, and reported every clock inactive) */
    {
        Instance *dead = fake_instance(SOCK_INVALID);
        g_log_calls = 0;
        CHECK(fmi3GetClock((fmi3Instance)dead, vr, 2, clk) == fmi3Error);
        CHECK(g_log_calls == 1 && strstr(g_last_log, "Event Mode") != NULL);
        CHECK(clk[0] == fmi3ClockActive);                       /* untouched */
        CHECK(fmi3SetClock((fmi3Instance)dead, vr, 2, clk) == fmi3Error);
        CHECK(g_log_calls == 2);
        free_instance(dead);
    }
    CHECK(fmi3GetClock(NULL, vr, 2, clk) == fmi3Error);
    CHECK(fmi3SetClock(NULL, vr, 2, clk) == fmi3Error);
    fmi3Boolean b1, b2, b3, b4, b5; fmi3Float64 t;
    /* Event Mode's, and not declared: both answered fmi3OK in any state */
    b1 = b5 = fmi3True;
    CHECK(fmi3UpdateDiscreteStates(NULL, &b1, &b2, &b3, &b4, &b5, &t) == fmi3Error && !b1 && !b5);
    {
        Instance *dead = fake_instance(SOCK_INVALID);
        g_log_calls = 0;
        CHECK(fmi3UpdateDiscreteStates((fmi3Instance)dead, &b1, &b2, &b3, &b4, &b5, &t) == fmi3Error);
        CHECK(g_log_calls == 1 && strstr(g_last_log, "Event Mode") != NULL);
        CHECK(fmi3EvaluateDiscreteStates((fmi3Instance)dead) == fmi3Error);
        CHECK(g_log_calls == 2 && strstr(g_last_log, "providesEvaluateDiscreteStates") != NULL);
        free_instance(dead);
    }
    CHECK(fmi3EnterContinuousTimeMode(NULL) == fmi3Error);
    CHECK(fmi3GetString(NULL, vr, 1, NULL, 1) == fmi3Error);
    Instance *dead = fake_instance(SOCK_INVALID);
    /* model exchange's: it used to answer fmi3OK and move the wrapper's clock */
    g_log_calls = 0;
    CHECK(fmi3SetTime((fmi3Instance)dead, 3.0) == fmi3Error && dead->time == 0.0);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "model-exchange") != NULL);
    CHECK(fmi3SetTime(NULL, 3.0) == fmi3Error);
    /* no structural parameters, so no Configuration Mode (both answered fmi3OK) */
    CHECK(fmi3EnterConfigurationMode((fmi3Instance)dead) == fmi3Error);
    CHECK(strstr(g_last_log, "no structural parameters") != NULL);
    CHECK(fmi3ExitConfigurationMode((fmi3Instance)dead) == fmi3Error);
    CHECK(fmi3EnterConfigurationMode(NULL) == fmi3Error);
    CHECK(fmi3SetDebugLogging((fmi3Instance)dead, fmi3True, 0, NULL) == fmi3OK && dead->logging_on);
    CHECK(fmi3Terminate(NULL) == fmi3Error);
    CHECK(fmi3Reset(NULL) == fmi3Error);
    free_instance(dead);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    /* The locale the environment names (LC_ALL / LC_NUMERIC): test_c_unit.py
     * runs this binary once more under a ',' decimal-point locale, where
     * every check below also checks that the wire's numbers ignore it. */
    setlocale(LC_ALL, "");
    printf("wrapper source: %s\n", MADDENING_FMU_C);
    printf("wrapper version: %s\n", MADDENING_FMU_VERSION);
    printf("decimal point in effect: %s\n", localeconv()->decimal_point);
    test_parse_values();
    test_parse_values_non_finite();
    test_read_endpoint();
    test_bridge_call_paths();
    test_send_failures();
    test_a_signal_during_a_receive_is_retried();
    test_get_set_step();
    test_binary_get_set();
    test_binary_fmu_state();
    test_oversize_reply_kills_the_connection();
    test_max_set_frame_fits_the_bridge_limit();
    test_get_request_respects_the_frame_limit();
    test_getters_refuse_values_their_type_cannot_hold();
    test_int64_setters_refuse_values_a_double_cannot_carry();
    test_set_fmu_state_counts_the_frame_header();
    test_read_timeout();
    test_numbers_are_written_and_read_in_the_c_locale();
    test_the_wrappers_clock_follows_set_fmu_state_and_reset();
    test_a_silent_sidecar_times_out();
    test_fmu_state();
    test_get_fmu_state_reuses_the_state_object_it_is_handed();
    test_the_fmu_state_functions_keep_to_fmi_ownership();
    test_free_instance_frees_the_states_that_are_still_live();
    test_the_fmi_state_machine();
    test_instantiate("deadbeef-0000-4000-8000-000000000001");
    test_misc_entry_points();
    /* Every allocation of the wrapper and of the tests above was freed:
     * the plain build's leak check (the sanitized one has LeakSanitizer). */
    free(g_seen); g_seen = NULL;
    printf("live allocations at exit: %ld\n", live_allocs());
    CHECK(live_allocs() == 0);
    printf("maddening_fmu unit tests: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
