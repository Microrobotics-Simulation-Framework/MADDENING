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

/* Run `body` on the client end while the fake server answers once; the
 * request the server saw is left in g_seen (g_seen_len bytes, flag bit
 * in g_seen_flag) for the caller to inspect. */
static char *g_seen = NULL;
static size_t g_seen_len = 0;
static int g_seen_flag = 0;
#define WITH_SERVER_X(reply_, reply_len_, advertised_, flag_, body) do {         \
    int sv[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);              \
    ServerArgs args = { sv[1], (reply_), (reply_len_), (advertised_), (flag_),    \
                        NULL, 0, 0, 0 };                                         \
    pthread_t th; pthread_create(&th, NULL, server_thread, &args);               \
    Instance *in = fake_instance(sv[0]);                                         \
    body;                                                                        \
    pthread_join(th, NULL);                                                      \
    if (in->sock != SOCK_INVALID) sock_close(sv[0]);   /* unless the wrapper dropped it */ \
    if (!args.closed) sock_close(sv[1]);                                         \
    free(g_seen); g_seen = args.seen ? args.seen : strdup("");                   \
    g_seen_len = args.seen_len; g_seen_flag = args.seen_flag;                    \
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
    CHECK(g_seen_flag == 0 && strcmp(g_seen, "{\"op\":\"get\",\"vr\":[7,9]}") == 0);  /* get stays JSON */
    /* narrower widths are widened from the same float64 wire form */
    fmi3Float32 g32[2];
    WITH_BINARY_SERVER((const char *)frame, n, {
        CHECK(fmi3GetFloat32((fmi3Instance)in, vr, 2, g32, 2) == fmi3OK);   /* parsed by flag, not mode */
        CHECK(g32[0] == 2.5f);
    });
    /* fewer values requested than sent is fine; more is an error */
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 1, g64, 1) == fmi3OK && g64[0] == 2.5);
    });
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
    /* exactly the limit is still read (the bridge may send that much) */
    {
        unsigned char *max = (unsigned char *)malloc(FRAME_MAX);
        size_t hl = bin_payload(max, "{\"ok\":true,\"n\":8388602,\"dtype\":\"f64\",\"p\":12}", NULL, 0) - 4;
        memset(max + 4 + hl, 0, FRAME_MAX - 4 - hl);
        CHECK(4 + hl + 8 * 8388602ul == FRAME_MAX);
        WITH_BINARY_SERVER((const char *)max, FRAME_MAX, {
            in->binary = 1;
            CHECK(fmi3GetFloat64((fmi3Instance)in, vr, 2, g64, 2) == fmi3OK && g64[0] == 0.0);
            CHECK(in->sock != SOCK_INVALID);
        });
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
        const char *hdr = "{\"op\":\"set\",\"vr\":[7,9,11],\"n\":3,\"dtype\":\"f64\"}";
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
    CHECK(g_seen_flag == 1 && g_seen_len == 4 + strlen("{\"op\":\"set\",\"vr\":[7,9],\"n\":2,\"dtype\":\"f64\"}") + 16);
    /* non-finite values never leave the process on the binary path either */
    fmi3Float64 bad[2] = { 1.0, INFINITY };
    Instance *dead = fake_instance(SOCK_INVALID);
    dead->binary = 1;
    g_log_calls = 0;
    CHECK(fmi3SetFloat64((fmi3Instance)dead, vr, 2, bad, 2) == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "non-finite") != NULL);
    /* a set larger than the frame limit is refused before any buffer grows
     * (and before any value is read: the count alone decides) */
    CHECK(do_set(dead, vr, 1, v64, (size_t)FRAME_MAX / 8 + 1) == fmi3Error);
    CHECK(dead->req_cap == 0 && strstr(g_last_log, "frame limit") != NULL);
    free_instance(dead);
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
    fmi3FMUState st = NULL;
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK);
    });
    CHECK(st != NULL && ((FmuState *)st)->n == 12 && memcmp(((FmuState *)st)->blob, blob, 12) == 0);
    /* serialize / deserialize keep the raw bytes */
    size_t sz = 0;
    CHECK(fmi3SerializedFMUStateSize(NULL, st, &sz) == fmi3OK && sz == 12);
    fmi3Byte buf[12];
    CHECK(fmi3SerializeFMUState(NULL, st, buf, 12) == fmi3OK && memcmp(buf, blob, 12) == 0);
    fmi3FMUState st2 = NULL;
    CHECK(fmi3DeserializeFMUState(NULL, buf, 12, &st2) == fmi3OK);
    /* set_state on the binary path is length-delimited: no base64 check */
    WITH_SERVER("{\"ok\":true}", 0, {
        in->binary = 1;
        CHECK(fmi3SetFMUState((fmi3Instance)in, st2) == fmi3OK);
    });
    CHECK(g_seen_flag == 1);
    {
        const char *hdr = "{\"op\":\"set_state\",\"n\":12}";
        size_t hl = strlen(hdr);
        CHECK(g_seen_len == 4 + hl + 12 && get_be32((const unsigned char *)g_seen) == hl);
        CHECK(memcmp(g_seen + 4, hdr, hl) == 0 && memcmp(g_seen + 4 + hl, blob, 12) == 0);
    }
    /* the same blob on a JSON-mode instance is refused (not base64) */
    Instance *dead = fake_instance(SOCK_INVALID);
    g_log_calls = 0;
    CHECK(fmi3SetFMUState((fmi3Instance)dead, st2) == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "not valid") != NULL);
    /* an oversize blob is refused before the request buffer grows */
    FmuState huge = { (char *)"x", (size_t)FRAME_MAX + 1 };
    dead->binary = 1;
    CHECK(fmi3SetFMUState((fmi3Instance)dead, &huge) == fmi3Error && dead->req_cap == 0);
    free_instance(dead);
    fmi3FreeFMUState(NULL, &st); fmi3FreeFMUState(NULL, &st2);
    /* count / raw length mismatch on get_state */
    n = bin_payload(frame, "{\"ok\":true,\"n\":11}", blob, 12);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3Error);
    });
    CHECK(st == NULL && strstr(g_last_log, "length mismatch") != NULL);
    n = bin_payload(frame, "{\"ok\":true}", blob, 12);
    WITH_BINARY_SERVER((const char *)frame, n, {
        in->binary = 1;
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3Error);
    });
    /* a JSON get_state reply still works on a binary-mode instance */
    WITH_SERVER("{\"ok\":true,\"state\":\"QUJDRA==\"}", 0, {
        in->binary = 1;
        CHECK(fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK);
    });
    CHECK(st != NULL && ((FmuState *)st)->n == 8);
    fmi3FreeFMUState(NULL, &st);
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
        hl = (size_t)snprintf(hdr, sizeof hdr, "{\"op\":\"set\",\"vr\":[%u],\"n\":%lu,\"dtype\":\"f64\"}",
                              (unsigned)vr[0], (unsigned long)n);
        if (4 + hl + 8 * n <= FRAME_MAX) break;
        --n;
    }
    CHECK(4 + hl + 8 * (n + 1) > FRAME_MAX);              /* n is the largest that fits */
    double *vals = (double *)calloc(n + 1, sizeof(double));
    WITH_SERVER("{\"ok\":true}", 0, {
        in->binary = 1;
        CHECK(do_set(in, vr, 1, vals, n) == fmi3OK);
    });
    CHECK(g_seen_flag == 1 && g_seen_len == 4 + hl + 8 * n && g_seen_len <= FRAME_MAX);
    CHECK(get_be32((const unsigned char *)g_seen) == hl && memcmp(g_seen + 4, hdr, hl) == 0);
    Instance *dead = fake_instance(SOCK_INVALID);
    dead->binary = 1;
    g_log_calls = 0;
    CHECK(do_set(dead, vr, 1, vals, n + 1) == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);   /* refused, not "closed" */
    /* the header counts: twenty 10-digit value references push the same
     * payload over the limit even with a few values fewer */
    g_log_calls = 0;
    CHECK(do_set(dead, vr, 20, vals, n - 10) == fmi3Error);
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
    /* and an absurd count or vr list is refused before any buffer grows */
    Instance *fresh = fake_instance(SOCK_INVALID);
    fresh->binary = 1;
    CHECK(do_set(fresh, vr, 1, vals, (size_t)FRAME_MAX / 8 + 1) == fmi3Error && fresh->req_cap == 0);
    CHECK(do_set(fresh, vr, (size_t)FRAME_MAX, vals, 1) == fmi3Error && fresh->req_cap == 0);
    free_instance(fresh);
    free_instance(dead);
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
    Instance *fresh = fake_instance(SOCK_INVALID);
    g_log_calls = 0;
    CHECK(do_get(fresh, vr, (size_t)FRAME_MAX / 2 + 1, out, 1) == fmi3Error);
    CHECK(fresh->req_cap == 0);                    /* refused before any buffer grew */
    CHECK(g_log_calls == 1 && strstr(g_last_log, "frame limit") != NULL);
    free_instance(fresh);
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

typedef struct { int listen_fd; const char *hello_reply; int accepted; char *hello_seen; } Listener;
static void *listener_thread(void *p) {
    Listener *L = (Listener *)p;
    int c = accept(L->listen_fd, NULL, NULL);
    if (c < 0) return NULL;
    L->accepted = 1;
    char *req = fake_exchange(c, L->hello_reply, 0);      /* hello */
    free(L->hello_seen); L->hello_seen = req;
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

    /* an old (protocol-1) bridge: its hello lacks "protocol" -> JSON everywhere */
    char reply[256]; snprintf(reply, sizeof reply, "{\"ok\":true,\"token\":\"%s\",\"model\":\"m\"}", good_token);
    Listener L = { fd, reply, 0, NULL };
    pthread_t th; pthread_create(&th, NULL, listener_thread, &L);
    fmi3Instance inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3True,
                                                    fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst != NULL);
    if (inst) {
        Instance *in = (Instance *)inst;
        CHECK(strcmp(in->instance_name, "i") == 0 && in->logging_on == fmi3True);
        CHECK(in->binary == 0);
        CHECK(fmi3EnterInitializationMode(inst, fmi3False, 0, 0.5, fmi3False, 0) == fmi3OK && in->time == 0.5);
        CHECK(fmi3ExitInitializationMode(inst) == fmi3OK);
        fmi3FreeInstance(inst);                          /* sends terminate */
    }
    pthread_join(th, NULL);
    CHECK(L.accepted == 1);
    /* the client always offers protocol 2 */
    CHECK(L.hello_seen != NULL && strcmp(L.hello_seen, "{\"op\":\"hello\",\"protocol\":2,\"binary\":true}") == 0);
    free(L.hello_seen);

    /* a protocol-2 bridge that confirms binary frames */
    char reply2[256];
    snprintf(reply2, sizeof reply2,
             "{\"ok\":true,\"token\":\"%s\",\"model\":\"m\",\"master_dt\":0.01,\"protocol\":2,\"binary\":true}",
             good_token);
    Listener Lb = { fd, reply2, 0, NULL };
    pthread_create(&th, NULL, listener_thread, &Lb);
    inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3False,
                                       fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst != NULL);
    if (inst) { CHECK(((Instance *)inst)->binary == 1); fmi3FreeInstance(inst); }
    pthread_join(th, NULL);
    free(Lb.hello_seen);

    /* protocol 2 announced but binary declined: JSON */
    snprintf(reply2, sizeof reply2,
             "{\"ok\":true,\"token\":\"%s\",\"protocol\":2,\"binary\":false}", good_token);
    Listener Lc = { fd, reply2, 0, NULL };
    pthread_create(&th, NULL, listener_thread, &Lc);
    inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3False,
                                       fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst != NULL);
    if (inst) { CHECK(((Instance *)inst)->binary == 0); fmi3FreeInstance(inst); }
    pthread_join(th, NULL);
    free(Lc.hello_seen);

    /* a bridge that refuses the protocol -> NULL */
    Listener Ld = { fd, "{\"ok\":false,\"error\":\"protocol 2 is not supported\"}", 0, NULL };
    pthread_create(&th, NULL, listener_thread, &Ld);
    g_log_calls = 0;
    inst = fmi3InstantiateCoSimulation("i", good_token, NULL, fmi3False, fmi3False,
                                       fmi3False, fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst == NULL && g_log_calls >= 1 && strstr(g_last_log, "not supported") != NULL);
    pthread_join(th, NULL);            /* its second exchange sees the client's EOF */
    free(Ld.hello_seen);

    /* token mismatch -> NULL, with a log line */
    Listener L2 = { fd, reply, 0, NULL };
    pthread_create(&th, NULL, listener_thread, &L2);
    g_log_calls = 0;
    inst = fmi3InstantiateCoSimulation("i", "wrong-token", NULL, fmi3False, fmi3False, fmi3False,
                                       fmi3False, NULL, 0, NULL, test_logger, NULL);
    CHECK(inst == NULL && g_log_calls >= 1 && strstr(g_last_log, "token") != NULL);
    shutdown(fd, SHUT_RDWR); sock_close(fd);
    pthread_join(th, NULL);
    free(L2.hello_seen);

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
    test_binary_get_set();
    test_binary_fmu_state();
    test_oversize_reply_kills_the_connection();
    test_max_set_frame_fits_the_bridge_limit();
    test_get_request_respects_the_frame_limit();
    test_fmu_state();
    test_instantiate("deadbeef-0000-4000-8000-000000000001");
    test_misc_entry_points();
    printf("maddening_fmu unit tests: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
