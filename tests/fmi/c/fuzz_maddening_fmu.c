/*
 * Fuzz harness for the FMU C wrapper's untrusted-input surface: the
 * framed reply from the sidecar (length header + JSON body, or a
 * binary-flagged frame with a random header-length / header / raw
 * combination), the JSON number parser, the binary header parser, the
 * hello negotiation and the value-reference / value arrays the importer
 * hands in.  Each iteration picks one FMI operation, preloads one reply
 * (well formed for that operation about a third of the time, otherwise
 * fuzzed) into a socketpair fake server and runs the operation; memory
 * errors are caught by ASan/UBSan (run via tests/fmi/test_c_unit.py).  A
 * deterministic xorshift PRNG makes every run reproducible from its seed.
 *
 * The harness is SELF-CHECKING: the wrapper is compiled with
 * MADDENING_FUZZ_COUNTERS, which counts how often each parser path was
 * reached, and the standalone `main` fails (exit 1) when any path was
 * never reached, so a broken fake server can no longer pass silently
 * (from 86dafe1 to the audit of 2026-09-16 the peer was closed before the
 * client sent its request: every exchange failed with EPIPE and no reply
 * was ever parsed).
 *
 *   ./fuzz <seed> <iterations>          standalone
 *   -DLIBFUZZER: exports LLVMFuzzerTestOneInput for clang -fsanitize=fuzzer
 */

#define MADDENING_FUZZ_COUNTERS 1
#include "../../../src/maddening/fmi/c/maddening_fmu.c"

#include <stdint.h>

static uint64_t g_rng = 88172645463325252ULL;
static uint64_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}

static void null_logger(fmi3InstanceEnvironment env, fmi3Status st, fmi3String cat, fmi3String msg) {
    (void)env; (void)st; (void)cat; (void)msg;
}

/* The fake sidecar's reply is written into the socketpair *before* the
 * client runs (a request + reply of at most 64 KiB fits the socket
 * buffers), so no thread is needed per iteration: deterministic, fast,
 * and no per-thread stacks for the sanitizers to track (a threaded
 * variant reached >7 GB RSS under libFuzzer on a CI runner).  The server
 * end is only HALF-closed (SHUT_WR): the client's request is still
 * accepted into the peer's buffer, so the client goes on to read the
 * reply; the EOF after it turns a short frame into a failed recv.  A
 * full shutdown (close_early) makes the send itself fail with EPIPE.
 * The caller closes both ends after the iteration. */
static void preload_reply(int srv, const unsigned char *body, size_t len,
                          size_t advertised, int flag, int close_early) {
    if (close_early) { shutdown(srv, SHUT_RDWR); return; }
    unsigned char h2[4];
    put_be32(h2, (unsigned long)advertised | (flag ? FRAME_BINARY : 0ul));
    send_all(srv, (const char *)h2, 4);
    send_all(srv, (const char *)body, len);
    shutdown(srv, SHUT_WR);
}

static const char *const PIECES[] = {
    "{\"ok\":true", "{\"ok\":false", ",\"values\":[", "]", ",", "1.5", "-0", "1e308", "-1e-320",
    "nan", "inf", "abc", "\"error\":\"", "\"state\":\"", "\"", "}", "{", " ", "0", "9999999999999999999999",
    "\"token\":\"", "\\u0000", "\n", "[", "]]", ",,", "1.", ".5", "0x10", "+3",
    ",\"protocol\":", "2", "1", "-2", ",\"binary\":true", ",\"binary\":false", "\"t\":",
};

static const char *const BIN_PIECES[] = {
    "{\"ok\":true", "{\"ok\":false", ",\"n\":", "0", "1", "2", "7", "16", "4294967296", "-1", "1e3", " ",
    ",\"dtype\":\"f64\"", ",\"dtype\":\"f32\"", ",\"error\":\"", "\"", "}", "{", "99999999999999999999",
    ",\"values\":[1]", ",\"state\":\"QUJDRA==\"", "\\u0000", ",", "5abc", "0x10",
};

/* Operation of one iteration (the reply generator knows it). */
enum { OP_GET = 0, OP_SET, OP_GET_STATE, OP_STEP, OP_HELLO, OP_SET_STATE, OP_COUNT };

/* A fuzzed binary payload: [u32 header_len][header][raw].  The header is
 * built from JSON-ish pieces (or is a consistent {"ok":true,"n":N,...}
 * that is then mutated); header_len is right most of the time, sometimes
 * larger than the payload, sometimes past the header cap; the raw part is
 * random bytes whose length only sometimes agrees with N. */
static size_t build_binary_reply(unsigned char *out, size_t cap) {
    char hdr[512];
    size_t hl = 0;
    if (rnd() % 2) {
        size_t k = rnd() % 12;
        for (size_t i = 0; i < k; ++i) {
            const char *pc = BIN_PIECES[rnd() % (sizeof BIN_PIECES / sizeof *BIN_PIECES)];
            size_t l = strlen(pc);
            if (hl + l >= sizeof hdr) break;
            memcpy(hdr + hl, pc, l); hl += l;
        }
    } else {
        unsigned long nv = (unsigned long)(rnd() % 20);
        const char *kind = (rnd() % 4 == 0) ? "" : ",\"dtype\":\"f64\"";
        hl = (size_t)snprintf(hdr, sizeof hdr, "{\"ok\":true,\"n\":%lu%s}", nv, kind);
        size_t flips = rnd() % 3;
        for (size_t i = 0; i < flips && hl; ++i) hdr[rnd() % hl] = (char)rnd();
    }
    size_t raw = rnd() % 200;
    if (rnd() % 4 == 0) raw = 8 * (rnd() % 20);            /* whole doubles */
    if (4 + hl + raw > cap) raw = cap - 4 - hl;
    unsigned long announced = (unsigned long)hl;
    switch (rnd() % 8) {
    case 0: announced = (unsigned long)(hl + raw + 1 + rnd() % 64); break;   /* past the payload */
    case 1: announced = HDR_MAX + rnd() % 8; break;                          /* past the cap */
    case 2: announced = (unsigned long)rnd(); break;                         /* anything */
    default: break;
    }
    put_be32(out, announced);
    memcpy(out + 4, hdr, hl);
    for (size_t i = 0; i < raw; ++i) out[4 + hl + i] = (unsigned char)rnd();
    return 4 + hl + raw;
}

/* A well-formed reply for `op` (F10 of the 2026-09-16 audit: the random
 * generator alone almost never makes the count and the raw length agree,
 * so the success paths of the binary parsers were never reached).  A
 * get reply carries at least `nvals` values; every raw double is a random
 * bit pattern (NaN payloads included: the copy must not care). */
static size_t build_good_reply(unsigned char *out, size_t cap, int op, size_t nvals, int *flag) {
    char hdr[128];
    size_t n = 0;
    *flag = 0;
    switch (op) {
    case OP_GET:
        if (rnd() % 3) {                                     /* binary form */
            size_t nv = nvals + rnd() % 4;
            size_t hl = (size_t)snprintf(hdr, sizeof hdr, "{\"ok\":true,\"n\":%lu,\"dtype\":\"f64\"}",
                                         (unsigned long)nv);
            put_be32(out, (unsigned long)hl);
            memcpy(out + 4, hdr, hl);
            for (size_t i = 0; i < 8 * nv; ++i) out[4 + hl + i] = (unsigned char)rnd();
            *flag = 1;
            return 4 + hl + 8 * nv;
        }
        n = (size_t)snprintf((char *)out, cap, "{\"ok\":true,\"values\":[");
        for (size_t i = 0; i < nvals + rnd() % 3; ++i)
            n += (size_t)snprintf((char *)out + n, cap - n, "%s%.17g", i ? "," : "",
                                  (double)(int64_t)rnd() / 4096.0);
        n += (size_t)snprintf((char *)out + n, cap - n, "]}");
        return n;
    case OP_GET_STATE:
        if (rnd() % 3) {                                     /* binary form: raw npz-ish bytes */
            size_t L = rnd() % 300;
            size_t hl = (size_t)snprintf(hdr, sizeof hdr, "{\"ok\":true,\"n\":%lu}", (unsigned long)L);
            put_be32(out, (unsigned long)hl);
            memcpy(out + 4, hdr, hl);
            for (size_t i = 0; i < L; ++i) out[4 + hl + i] = (unsigned char)rnd();
            *flag = 1;
            return 4 + hl + L;
        }
        return (size_t)snprintf((char *)out, cap, "{\"ok\":true,\"state\":\"%.*s\"}",
                                (int)(4 * (rnd() % 8)), "QUJDRA==QUJDRA==QUJDRA==QUJDRA==");
    case OP_STEP:
        return (size_t)snprintf((char *)out, cap, "{\"ok\":true,\"t\":%.17g}", (double)(rnd() % 1000) / 7.0);
    case OP_HELLO:
        switch (rnd() % 4) {
        case 0: return (size_t)snprintf((char *)out, cap, "{\"ok\":true,\"token\":\"t\",\"model\":\"m\"}");
        case 1: return (size_t)snprintf((char *)out, cap,
                                        "{\"ok\":true,\"token\":\"t\",\"protocol\":2,\"binary\":false}");
        case 2: return (size_t)snprintf((char *)out, cap,
                                        "{\"ok\":true,\"token\":\"t\",\"protocol\":%lu,\"binary\":true}",
                                        (unsigned long)(rnd() % 5));
        default: return (size_t)snprintf((char *)out, cap,
                                         "{\"ok\":true,\"token\":\"t\",\"master_dt\":0.01,\"protocol\":2,\"binary\":true}");
        }
    default:                                                 /* set / set_state */
        return (size_t)snprintf((char *)out, cap, "{\"ok\":true}");
    }
}

static size_t build_reply(unsigned char *out, size_t cap, int *flag, int op, size_t nvals) {
    size_t n = 0;
    *flag = 0;
    if (rnd() % 3 == 0) return build_good_reply(out, cap, op, nvals, flag);
    if (rnd() % 3 == 0) {
        *flag = 1;
        if (rnd() % 8 == 0) {                        /* a JSON body under the binary flag */
            n = rnd() % 64;
            for (size_t i = 0; i < n; ++i) out[i] = (unsigned char)rnd();
            return n;
        }
        return build_binary_reply(out, cap);
    }
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
        n = 1000 + rnd() % 30000;
        if (n > cap) n = cap;
        memset(out, (int)('0' + rnd() % 10), n);
        memcpy(out, "{\"ok\":true,\"values\":[", 22);
        break;
    }
    }
    return n;
}

static void one_iteration(unsigned char *buf, size_t cap) {
    int op = (int)(rnd() % OP_COUNT);
    size_t nvr = rnd() % 8, nvals = rnd() % 16;
    int flag;
    size_t n = build_reply(buf, cap, &flag, op, nvals);
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) return;
    /* Bound the reply so it always fits the socket buffer un-read. */
    int bufsz = 1 << 17;
    setsockopt(sv[1], SOL_SOCKET, SO_SNDBUF, &bufsz, sizeof bufsz);
    setsockopt(sv[0], SOL_SOCKET, SO_RCVBUF, &bufsz, sizeof bufsz);
    size_t advertised = (rnd() % 8 == 0) ? n + rnd() % 64 : n;   /* sometimes lie about the length */
    if (rnd() % 32 == 0) advertised = FRAME_MAX + 1 + rnd() % 1024;  /* over the cap, either kind */
    preload_reply(sv[1], buf, n, advertised, flag, (rnd() % 16 == 0));

    Instance *in = (Instance *)calloc(1, sizeof *in);
    in->sock = sv[0]; in->log = null_logger;
    in->binary = (int)(rnd() % 2);                 /* half the runs negotiated protocol 2 */
    fmi3ValueReference vr[8]; double vals[16];
    for (size_t i = 0; i < 8; ++i) vr[i] = (fmi3ValueReference)rnd();
    for (size_t i = 0; i < 16; ++i) vals[i] = (double)(int64_t)rnd() / 1e6;
    switch (op) {
    case OP_GET: do_get(in, vr, nvr, vals, nvals); break;
    case OP_SET: do_set(in, vr, nvr, vals, nvals); break;
    case OP_SET_STATE: { /* set_state with importer-supplied bytes of either encoding */
              unsigned char blob[64]; size_t bl = rnd() % 64;
              for (size_t i = 0; i < bl; ++i) blob[i] = (rnd() % 2) ? (unsigned char)rnd() : 'A';
              fmi3FMUState st = NULL;
              if (fmi3DeserializeFMUState(NULL, blob, bl, &st) == fmi3OK) {
                  fmi3SetFMUState((fmi3Instance)in, st);
                  fmi3FreeFMUState(NULL, &st);
              }
              break; }
    case OP_GET_STATE: { fmi3FMUState st = NULL;
              if (fmi3GetFMUState((fmi3Instance)in, &st) == fmi3OK) {
                  size_t sz = 0;
                  if (fmi3SerializedFMUStateSize(NULL, st, &sz) == fmi3OK) {
                      fmi3Byte *b = (fmi3Byte *)malloc(sz + 1);
                      if (fmi3SerializeFMUState(NULL, st, b, sz) == fmi3OK) {
                          fmi3FMUState st2 = NULL;
                          if (fmi3DeserializeFMUState(NULL, b, sz, &st2) == fmi3OK)
                              fmi3FreeFMUState(NULL, &st2);
                      }
                      free(b);
                  }
              }
              fmi3FreeFMUState(NULL, &st); break; }
    case OP_STEP: { fmi3Boolean e, t, r; fmi3Float64 l;
              fmi3DoStep((fmi3Instance)in, (double)(rnd() % 1000) / 7.0, 1e-3, fmi3False, &e, &t, &r, &l);
              break; }
    default:  /* hello: the negotiation, then the number parser on whatever came back */
              if (bridge_call(in, "{\"op\":\"hello\",\"protocol\":2,\"binary\":true}") == fmi3OK)
                  in->binary = hello_negotiated_binary(in);
              parse_values(in, vals, nvals);
              break;
    }
    /* a dropped connection (over-limit / truncated reply) must stay dead */
    if (in->sock == SOCK_INVALID) {
        fmi3Boolean e, t, r; fmi3Float64 l;
        if (fmi3DoStep((fmi3Instance)in, 0.0, 1e-3, fmi3False, &e, &t, &r, &l) == fmi3OK) abort();
    } else {
        sock_close(sv[0]);
    }
    sock_close(sv[1]);
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
    /* Every path must have been reached, or the harness is not fuzzing
     * what it claims to (a peer closed too early, a generator that never
     * produces a well-formed frame, ...). */
    const struct { const char *name; unsigned long count; } paths[] = {
        { "replies_read", fuzz_counters.replies_read },
        { "binary_replies", fuzz_counters.binary_replies },
        { "parse_values", fuzz_counters.parse_values },
        { "parse_values_ok", fuzz_counters.parse_values_ok },
        { "parse_binary_values", fuzz_counters.parse_binary_values },
        { "parse_binary_ok", fuzz_counters.parse_binary_ok },
        { "hello_negotiated", fuzz_counters.hello_negotiated },
        { "hello_binary", fuzz_counters.hello_binary },
        { "raw_state", fuzz_counters.raw_state },
        { "raw_state_ok", fuzz_counters.raw_state_ok },
        { "conn_dropped", fuzz_counters.conn_dropped },
    };
    int missing = 0;
    printf("fuzz paths:");
    for (size_t i = 0; i < sizeof paths / sizeof *paths; ++i) {
        printf(" %s=%lu", paths[i].name, paths[i].count);
        if (paths[i].count == 0) ++missing;
    }
    printf("\n");
    if (missing) {
        for (size_t i = 0; i < sizeof paths / sizeof *paths; ++i)
            if (paths[i].count == 0)
                fprintf(stderr, "fuzz: path %s was never reached: the harness is not exercising the parser\n",
                        paths[i].name);
        printf("fuzz: seed=%llu iterations=%ld FAILED (%d paths unreached)\n",
               (unsigned long long)seed, iters, missing);
        return 1;
    }
    printf("fuzz: seed=%llu iterations=%ld ok\n", (unsigned long long)seed, iters);
    return 0;
}
#endif
