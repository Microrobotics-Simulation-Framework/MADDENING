/*
 * Fuzz harness for the FMU C wrapper's untrusted-input surface: the
 * framed reply from the sidecar (length header + JSON body), the JSON
 * number parser, and the value-reference / value arrays the importer
 * hands in.  Each iteration feeds one fuzzed reply through a socketpair
 * fake server into bridge_call / parse_values / the getters and the FMU
 * state path; memory errors are caught by ASan/UBSan (run via
 * tests/fmi/test_c_unit.py).  A deterministic xorshift PRNG makes every
 * run reproducible from its seed.
 *
 *   ./fuzz <seed> <iterations>          standalone
 *   -DLIBFUZZER: exports LLVMFuzzerTestOneInput for clang -fsanitize=fuzzer
 */

#include "../../../src/maddening/fmi/c/maddening_fmu.c"

#include <pthread.h>
#include <stdint.h>

static uint64_t g_rng = 88172645463325252ULL;
static uint64_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}

static void null_logger(fmi3InstanceEnvironment env, fmi3Status st, fmi3String cat, fmi3String msg) {
    (void)env; (void)st; (void)cat; (void)msg;
}

typedef struct { int s; const unsigned char *body; size_t len; size_t advertised; int close_early; } Srv;

static void *serve_one(void *p) {
    Srv *a = (Srv *)p;
    unsigned char head[4];
    if (recv_all(a->s, (char *)head, 4)) { sock_close(a->s); return NULL; }
    size_t n = ((size_t)head[0] << 24) | ((size_t)head[1] << 16) | ((size_t)head[2] << 8) | head[3];
    char *req = (char *)malloc(n + 1);
    if (recv_all(a->s, req, n)) { free(req); sock_close(a->s); return NULL; }
    free(req);
    if (a->close_early) { sock_close(a->s); return NULL; }
    unsigned char h2[4] = { (unsigned char)(a->advertised >> 24), (unsigned char)(a->advertised >> 16),
                            (unsigned char)(a->advertised >> 8), (unsigned char)a->advertised };
    send_all(a->s, (const char *)h2, 4);
    send_all(a->s, (const char *)a->body, a->len);
    sock_close(a->s);
    return NULL;
}

static const char *const PIECES[] = {
    "{\"ok\":true", "{\"ok\":false", ",\"values\":[", "]", ",", "1.5", "-0", "1e308", "-1e-320",
    "nan", "inf", "abc", "\"error\":\"", "\"state\":\"", "\"", "}", "{", " ", "0", "9999999999999999999999",
    "\"token\":\"", "\\u0000", "\n", "[", "]]", ",,", "1.", ".5", "0x10", "+3",
};

static size_t build_reply(unsigned char *out, size_t cap) {
    size_t n = 0;
    switch (rnd() % 4) {
    case 0: {                                   /* random bytes */
        n = rnd() % (cap / 8);
        for (size_t i = 0; i < n; ++i) out[i] = (unsigned char)rnd();
        break;
    }
    case 1: {                                   /* JSON-ish pieces */
        size_t k = rnd() % 24;
        for (size_t i = 0; i < k; ++i) {
            const char *pc = PIECES[rnd() % (sizeof PIECES / sizeof *PIECES)];
            size_t l = strlen(pc);
            if (n + l >= cap) break;
            memcpy(out + n, pc, l); n += l;
        }
        break;
    }
    case 2: {                                   /* valid reply, then mutated */
        int nv = (int)(rnd() % 6);
        n = (size_t)snprintf((char *)out, cap, "{\"ok\":true,\"values\":[");
        for (int i = 0; i < nv; ++i)
            n += (size_t)snprintf((char *)out + n, cap - n, "%s%.17g", i ? "," : "",
                                  (double)(int64_t)rnd() / 4096.0);
        n += (size_t)snprintf((char *)out + n, cap - n, "],\"state\":\"%.*s\"}", (int)(rnd() % 40), "QUJDRA==QUJDRA==QUJDRA==QUJDRA==QUJDRA==");
        size_t flips = rnd() % 4;
        for (size_t i = 0; i < flips && n; ++i) out[rnd() % n] = (unsigned char)rnd();
        break;
    }
    default: {                                  /* long run */
        n = 1000 + rnd() % 60000;
        if (n > cap) n = cap;
        memset(out, (int)('0' + rnd() % 10), n);
        memcpy(out, "{\"ok\":true,\"values\":[", 22);
        break;
    }
    }
    return n;
}

static void one_iteration(unsigned char *buf, size_t cap) {
    size_t n = build_reply(buf, cap);
    Srv a; int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) return;
    a.s = sv[1]; a.body = buf; a.len = n;
    a.advertised = (rnd() % 8 == 0) ? n + rnd() % 64 : n;   /* sometimes lie about the length */
    a.close_early = (rnd() % 16 == 0);
    pthread_t th; pthread_create(&th, NULL, serve_one, &a);

    Instance *in = (Instance *)calloc(1, sizeof *in);
    in->sock = sv[0]; in->log = null_logger;
    fmi3ValueReference vr[8]; double vals[16];
    size_t nvr = rnd() % 8, nvals = rnd() % 16;
    for (size_t i = 0; i < 8; ++i) vr[i] = (fmi3ValueReference)rnd();
    for (size_t i = 0; i < 16; ++i) vals[i] = (double)(int64_t)rnd() / 1e6;
    switch (rnd() % 5) {
    case 0: do_get(in, vr, nvr, vals, nvals); break;
    case 1: do_set(in, vr, nvr, vals, nvals); break;
    case 2: { fmi3FMUState st = NULL;
              if (fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK) {
                  size_t sz; fmi3SerializedFMUStateSize(NULL, st, &sz);
                  fmi3Byte *b = (fmi3Byte *)malloc(sz + 1);
                  fmi3SerializeFMUState(NULL, st, b, sz);
                  fmi3FMUState st2; fmi3DeserializeFMUState(NULL, b, sz, &st2);
                  fmi3FreeFMUState(NULL, &st2); free(b);
              }
              fmi3FreeFMUState(NULL, &st); break; }
    case 3: { fmi3Boolean e, t, r; fmi3Float64 l;
              fmi3DoStep((fmi3Instance)in, (double)(rnd() % 1000) / 7.0, 1e-3, fmi3False, &e, &t, &r, &l);
              break; }
    default: bridge_call(in, "{\"op\":\"hello\"}"); parse_values(in, vals, nvals); break;
    }
    pthread_join(th, NULL);
    sock_close(sv[0]);
    free(in->req); free(in->resp); free(in);
}

#ifdef LIBFUZZER
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    static unsigned char buf[1 << 16];
    if (size == 0) return 0;
    g_rng = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < size; ++i) g_rng = g_rng * 31 + data[i];
    if (!g_rng) g_rng = 1;
    one_iteration(buf, sizeof buf);
    return 0;
}
#else
int main(int argc, char **argv) {
    uint64_t seed = argc > 1 ? strtoull(argv[1], NULL, 10) : 1;
    long iters = argc > 2 ? strtol(argv[2], NULL, 10) : 2000;
    g_rng = seed ? seed : 1;
    static unsigned char buf[1 << 16];
    for (long i = 0; i < iters; ++i) one_iteration(buf, sizeof buf);
    printf("fuzz: seed=%llu iterations=%ld ok\n", (unsigned long long)seed, iters);
    return 0;
}
#endif
