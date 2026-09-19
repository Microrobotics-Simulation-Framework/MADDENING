/* Targeted probe of the request-building side of the FMU wrapper:
 * buffer sizing for %.17g, extreme doubles, big nvr, FMU-state framing. */
#define MADDENING_FUZZ_COUNTERS 1
#include "../../../../../home/nick/MSF/msf/MADDENING-wt/audit/params-io/src/maddening/fmi/c/maddening_fmu.c"
#include <stdint.h>
#include <float.h>
#include <sys/socket.h>

static void null_logger(fmi3InstanceEnvironment e, fmi3Status s, fmi3String c, fmi3String m) {
    (void)e;(void)s;(void)c;(void)m;
}

/* 1. widest %.17g any double can produce */
static void widest_g(void) {
    size_t worst = 0; double warg = 0;
    char b[512];
    double cands[] = {
        DBL_MAX, -DBL_MAX, DBL_MIN, -DBL_MIN, 5e-324, -5e-324,
        -1.2345678901234567e-308, -1.2345678901234567e+308,
        -0.00012345678901234567, -1234567890123456.7, -12345678901234567.0,
        -0.000123456789012345678, 1.0/3.0, -1.0/3.0, -2.2250738585072014e-308,
    };
    for (size_t i = 0; i < sizeof cands/sizeof *cands; ++i) {
        int n = snprintf(b, sizeof b, "%.17g", cands[i]);
        if ((size_t)n > worst) { worst = (size_t)n; warg = cands[i]; }
    }
    /* plus a random sweep over bit patterns */
    uint64_t s = 0x12345678ULL;
    for (long i = 0; i < 20000; ++i) {
        s ^= s<<13; s ^= s>>7; s ^= s<<17;
        double d; memcpy(&d, &s, 8);
        if (isnan(d) || isinf(d)) continue;
        int n = snprintf(b, sizeof b, "%.17g", d);
        if ((size_t)n > worst) { worst = (size_t)n; warg = d; }
    }
    printf("widest %%.17g = %zu chars (e.g. %.17g); do_set budgets 32 per value\n", worst, warg);
}

static Instance *mk(int *sv, int binary) {
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    int bufsz = 1<<20;
    setsockopt(sv[1], SOL_SOCKET, SO_SNDBUF, &bufsz, sizeof bufsz);
    setsockopt(sv[0], SOL_SOCKET, SO_RCVBUF, &bufsz, sizeof bufsz);
    Instance *in = (Instance*)calloc(1, sizeof *in);
    in->sock = sv[0]; in->log = null_logger; in->binary = binary;
    return in;
}

/* 2. do_set (JSON) with the widest doubles and the max value count that fits */
static void set_extremes(void) {
    int sv[2]; Instance *in = mk(sv, 0);
    const char *ok = "{\"ok\":true}";
    unsigned char h[4]; put_be32(h, (unsigned long)strlen(ok));
    size_t N = 200000;
    double *v = (double*)malloc(N * sizeof *v);
    fmi3ValueReference *vr = (fmi3ValueReference*)malloc(N * sizeof *vr);
    for (size_t i = 0; i < N; ++i) {
        v[i] = (i % 4 == 0) ? -DBL_MAX : (i % 4 == 1) ? -5e-324
             : (i % 4 == 2) ? -1.2345678901234567e-308 : -0.00012345678901234567;
        vr[i] = 0xFFFFFFFFu;
    }
    (void)h; (void)ok;
    in->sock = SOCK_INVALID;          /* build the frame, never block sending it */
    fmi3Status st = do_set(in, vr, N, v, N);
    printf("do_set JSON  nvr=%zu nvalues=%zu -> status %d, frame %zu bytes (cap %zu)\n",
           N, N, (int)st, strlen(in->req), in->req_cap);
    free(v); free(vr); close(sv[0]); close(sv[1]); free(in->req); free(in->resp); free(in);
}

/* 3. do_get with an nvr whose request frame exceeds FRAME_MAX -- do_set
 *    refuses such a frame, do_get has no equivalent check. */
static void get_oversize(void) {
    int sv[2]; Instance *in = mk(sv, 0);
    size_t N = 7000000;                      /* ~11 bytes each -> > 64 MiB */
    fmi3ValueReference *vr = (fmi3ValueReference*)malloc(N * sizeof *vr);
    for (size_t i = 0; i < N; ++i) vr[i] = 0xFFFFFFFFu;
    double out[1];
    in->sock = SOCK_INVALID;
    fmi3Status st = do_get(in, vr, N, out, 0);
    size_t framelen = in->req ? strlen(in->req) : 0;
    printf("do_get  nvr=%zu -> status %d, request frame %zu bytes, FRAME_MAX=%lu, over=%s\n",
           N, (int)st, framelen, FRAME_MAX, framelen > FRAME_MAX ? "YES" : "no");
    free(vr); close(sv[0]); close(sv[1]); free(in->req); free(in->resp); free(in);
}

/* 4. fmi3SetFMUState JSON path with a blob of exactly the base64 alphabet */
static void set_state_json(void) {
    int sv[2]; Instance *in = mk(sv, 0);
    const char *ok = "{\"ok\":true}";
    unsigned char h[4]; put_be32(h, (unsigned long)strlen(ok));
    (void)h; (void)ok;
    in->sock = SOCK_INVALID;
    size_t n = 1u<<20;
    unsigned char *blob = (unsigned char*)malloc(n);
    for (size_t i = 0; i < n; ++i) blob[i] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[i % 65];
    fmi3FMUState st = NULL;
    fmi3DeserializeFMUState(NULL, blob, n, &st);
    fmi3Status s = fmi3SetFMUState((fmi3Instance)in, st);
    printf("fmi3SetFMUState JSON blob=%zu -> status %d, req strlen=%zu cap=%zu\n",
           n, (int)s, strlen(in->req), in->req_cap);
    fmi3FreeFMUState(NULL, &st);
    free(blob); close(sv[0]); close(sv[1]); free(in->req); free(in->resp); free(in);
}

/* 5. read_endpoint with a hostile MADDENING_FMU_ENDPOINT */
static void endpoint_probe(void) {
    char host[256]; int port = 0;
    const char *cases[] = {
        "127.0.0.1:5555", "[::1]:5555", ":5555", "127.0.0.1:", "127.0.0.1:0",
        "127.0.0.1:99999999999999999999", "[]:1", "a:1:2:3", "]:[1",
    };
    for (size_t i = 0; i < sizeof cases/sizeof *cases; ++i) {
        setenv("MADDENING_FMU_ENDPOINT", cases[i], 1);
        memset(host, 0, sizeof host); port = -1;
        int rc = read_endpoint(NULL, host, sizeof host, &port);
        printf("  endpoint %-34s -> rc=%d host=%-20s port=%d\n", cases[i], rc, host, port);
    }
    /* a 400-char host must not overflow the 256-byte buffer */
    char big[600]; memset(big, 'a', 500); strcpy(big + 500, ":9"); big[502] = 0;
    setenv("MADDENING_FMU_ENDPOINT", big, 1);
    int rc = read_endpoint(NULL, host, sizeof host, &port);
    printf("  endpoint <500-char host>            -> rc=%d port=%d\n", rc, port);
    unsetenv("MADDENING_FMU_ENDPOINT");
}

int main(void) {
    widest_g();
    set_extremes();
    get_oversize();
    set_state_json();
    printf("read_endpoint:\n"); endpoint_probe();
    printf("done\n");
    return 0;
}
