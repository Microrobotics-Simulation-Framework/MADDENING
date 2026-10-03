/*
 * MADDENING FMU C wrapper (FMI 3.0 co-simulation).
 *
 * The FMU binary owns no simulation state: every FMI call is marshalled
 * into a small message and forwarded over a TCP socket to a Python
 * sidecar (maddening.fmi.tcp_bridge.FmuTcpBridge) that holds the
 * JAX-JITted graph.  Wire format: 4-byte big-endian length prefix + one
 * frame, both directions.  Bit 31 of the prefix marks a *binary* frame
 * ("[u32 BE header_len][header JSON][raw bytes]"), the low 31 bits are
 * the payload length; a frame without the bit is one UTF-8 JSON object.
 *
 * The wrapper offers protocol 2 at hello ({"op":"hello","protocol":2,
 * "binary":true}).  A bridge that answers with "protocol":2 and
 * "binary":true then gets bulk values as raw little-endian float64
 * (set requests, get replies) and FMU-state blobs as raw bytes (no
 * base64, no %.17g / strtod).  A bridge whose hello reply lacks
 * "protocol" is protocol 1: everything stays JSON, as before.
 *
 * On the JSON path a non-finite value arrives as a quoted token
 * ("NaN", "Infinity", "-Infinity"), because the bare tokens are not
 * JSON; parse_values steps over the quotes and lets C99 strtod read
 * the token.  Bare tokens are still accepted.  Non-finite values are
 * never *sent*: do_set refuses them before the request is built.
 *
 * Every get and set request names the FMI type of the function that made
 * it ("type":"Float32", ...), and the bridge refuses a variable of another
 * type: FMI 3.0 has a variable accessed only through the fmi3Get/Set
 * function of its own type, and this wrapper carries every width as a
 * double.  A getter also refuses a reply value its C type cannot hold (a
 * fraction, NaN or an out-of-range number for an integer, anything but
 * 0/1 for a Boolean), so no conversion here is undefined behaviour or a
 * silent truncation; an Int64/UInt64 setter refuses a value a double
 * cannot carry exactly.
 *
 * The endpoint is read from "<resourcePath>/endpoint.txt" ("host:port"),
 * or from the MADDENING_FMU_ENDPOINT environment variable.
 *
 * Deadlines.  MADDENING_FMU_TIMEOUT (seconds, decimal; default 600) bounds
 * how long the wrapper waits on the sidecar: every receive and send on the
 * connection times out after that much silence, and a timeout closes the
 * connection, so the call and every later one return fmi3Error.  Connecting
 * (on Linux, where SO_SNDTIMEO bounds connect) and the hello exchange get
 * the smaller of that and 30 s: the bridge answers a hello at once, so an
 * endpoint that accepts and never answers fails fmi3InstantiateCoSimulation
 * in 30 s instead of blocking it for ever.  Ten minutes covers a first
 * doStep that compiles a large graph and a step of many graph steps (the
 * bridge caps one request at 100000 by default); raise it for a longer
 * step, or set 0 to wait for ever (the behaviour before 0.4.0).  A value
 * that is not a number from 0 to 1e6 fails instantiation.  The deadline is
 * on silence, not on a whole reply: a sidecar that keeps sending is never
 * cut off.
 *
 * Numbers on the JSON path are written and read in the C locale's
 * LC_NUMERIC, whatever locale the importer runs under: %.17g and strtod
 * follow the process's LC_NUMERIC, so under a locale whose decimal
 * separator is ',' (any GUI tool that calls setlocale(LC_ALL, "") on a
 * Dutch or German desktop) every doStep sent "dt":0,01, which is not JSON,
 * and failed.  Each instance holds a C LC_NUMERIC locale (newlocale on
 * POSIX, _create_locale on Windows) and formats and parses through it
 * (uselocale around the call on POSIX, the _l functions on Windows).
 *
 * State machine.  The wrapper holds FMI 3.0's co-simulation states for an
 * FMU without Event Mode or structural parameters, and refuses a call its
 * state does not allow, with fmi3Error, a log message and nothing sent:
 *
 *   Instantiated  --fmi3EnterInitializationMode-->  Initialization Mode
 *   Initialization Mode  --fmi3ExitInitializationMode-->  Step Mode
 *   Step Mode  --fmi3Terminate-->  Terminated
 *   any state  --fmi3Reset-->  Instantiated
 *
 * fmi3DoStep is allowed in Step Mode only; fmi3Get* in Initialization Mode,
 * Step Mode and Terminated; fmi3Set* in Instantiated, Initialization Mode
 * and Step Mode (a zero-length call too).  fmi3EnterStepMode is always
 * refused: FMI 3.0 allows it only from Event Mode, and without Event Mode
 * fmi3ExitInitializationMode enters Step Mode itself.  So are
 * fmi3EnterConfigurationMode / fmi3ExitConfigurationMode (the FMU has no
 * structural parameters), fmi3SetTime (model exchange),
 * fmi3EvaluateDiscreteStates (not declared in the description), and
 * fmi3UpdateDiscreteStates, fmi3GetClock and fmi3SetClock, which FMI 3.0
 * allows only in Event Mode (the clocks are constant-interval, their ticks
 * implied by time).  The FMU-state
 * functions are allowed in every state and leave it as it is.  Until
 * 0.4.0's fix fmi3DoStep before fmi3EnterInitializationMode advanced the
 * model, and fmi3ExitInitializationMode, fmi3EnterStepMode and a
 * zero-length fmi3Set* answered fmi3OK after fmi3Terminate.  The bridge
 * keeps its own check of the Terminated state (it refuses doStep, set and
 * initialize until fmi3Reset).  The wrapper's own clock, which fmi3DoStep reports as
 * lastSuccessfulTime when it fails, follows fmi3SetFMUState (the bridge's
 * reply carries the restored time) and fmi3Reset (only once the reset
 * succeeded).
 *
 * Only libc and the FMI 3.0 headers are needed to build:
 *   cc -shared -fPIC -O2 -I<fmi3 headers> maddening_fmu.c -o maddening_fmu.so
 * (see maddening.fmi.package.build_fmu_binary).
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
/* getaddrinfo / ssize_t / strdup under -std=c11 -pedantic */
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE 1
#endif

#include <errno.h>
#include <float.h>
#include <locale.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__APPLE__)
#  include <xlocale.h>
#endif

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
   typedef SOCKET sock_t;
#  define SOCK_INVALID INVALID_SOCKET
#  define sock_close closesocket
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
   typedef int sock_t;
#  define SOCK_INVALID (-1)
#  define sock_close close
#endif

#include "fmi3Functions.h"

#define MADDENING_FMU_VERSION "0.4.0-dev"
#define REQ_CAP (1u << 20)
#define FRAME_BINARY 0x80000000ul            /* bit 31 of the length prefix */
#define FRAME_LEN_MASK 0x7FFFFFFFul
#define FRAME_MAX (64ul * 1024ul * 1024ul)   /* the bridge's frame limit, both
                                                directions and both frame kinds */
#define HDR_MAX 4096                         /* a binary reply's JSON header */
#define REPLY_TIMEOUT_DEFAULT_S 600.0        /* MADDENING_FMU_TIMEOUT unset */
#define REPLY_TIMEOUT_MAX_S 1.0e6            /* and its largest accepted value */
#define HANDSHAKE_TIMEOUT_S 30.0             /* connect + hello, at most */

/* Path counters for the fuzz harness (tests/fmi/c/fuzz_maddening_fmu.c),
 * which defines MADDENING_FUZZ_COUNTERS before including this file so it
 * can prove the parsers were reached.  Compiled out of the shipped FMU. */
#ifdef MADDENING_FUZZ_COUNTERS
static struct {
    unsigned long replies_read, binary_replies, parse_values, parse_values_ok,
                  parse_binary_values, parse_binary_ok, hello_negotiated, hello_binary,
                  raw_state, raw_state_ok, conn_dropped;
} fuzz_counters;
#  define FUZZ_COUNT(field) ((void)++fuzz_counters.field)
#else
#  define FUZZ_COUNT(field) ((void)0)
#endif

typedef struct {
    sock_t sock;
    char instance_name[256];
    fmi3InstanceEnvironment env;
    fmi3LogMessageCallback log;
    fmi3Boolean logging_on;
    double time;
    char *req;      /* request scratch buffer      */
    size_t req_cap;
    char *resp;     /* last response (NUL-terminated) */
    size_t resp_cap;
    int binary;     /* hello negotiated protocol 2 with binary frames */
    int resp_binary;       /* the last reply was a binary frame */
    char hdr[HDR_MAX];     /* its JSON header, NUL-terminated */
    const char *raw;       /* its raw part (points into resp) */
    size_t raw_len;
    double timeout_s;      /* MADDENING_FMU_TIMEOUT; 0 = no deadline */
    int phase;             /* FMI 3.0 state (Phase); 0, as calloc leaves it, is
                              Instantiated */
    int c_numeric_ready;   /* c_numeric holds a C LC_NUMERIC locale */
#ifdef _WIN32
    _locale_t c_numeric;
#else
    locale_t c_numeric;
#endif
} Instance;

/* ------------------------------------------------- FMI 3.0 state machine */

/* The co-simulation states of an FMU without Event Mode or structural
 * parameters (see the State machine note at the top of this file). */
typedef enum {
    PHASE_INSTANTIATED = 0,
    PHASE_INITIALIZATION_MODE = 1,
    PHASE_STEP_MODE = 2,
    PHASE_TERMINATED = 3
} Phase;

#define IN_INSTANTIATED   (1u << PHASE_INSTANTIATED)
#define IN_INITIALIZATION (1u << PHASE_INITIALIZATION_MODE)
#define IN_STEP_MODE      (1u << PHASE_STEP_MODE)
#define IN_TERMINATED     (1u << PHASE_TERMINATED)
#define ALLOWED_GET (IN_INITIALIZATION | IN_STEP_MODE | IN_TERMINATED)
#define ALLOWED_SET (IN_INSTANTIATED | IN_INITIALIZATION | IN_STEP_MODE)

static const char *phase_name(int phase) {
    switch (phase) {
    case PHASE_INSTANTIATED: return "Instantiated";
    case PHASE_INITIALIZATION_MODE: return "Initialization Mode";
    case PHASE_STEP_MODE: return "Step Mode";
    case PHASE_TERMINATED: return "Terminated";
    default: return "unknown";
    }
}

/* 1 if the instance's state allows `fn` (`allowed` is a mask of IN_*
 * bits, `where` says which states in words); else logs and returns 0, and
 * the caller returns fmi3Error with nothing sent.  An unknown state is
 * allowed nothing. */
static int phase_allows(Instance *in, unsigned allowed, const char *fn, const char *where);

/* ---------------------------------------------------- C-locale numbers */

/* The instance's C LC_NUMERIC locale, made on first use (so an Instance
 * built by calloc -- the unit tests' -- gets one too).  0 if the system
 * cannot make one, which instantiation refuses. */
static int c_numeric_ready(Instance *in) {
    if (in->c_numeric_ready) return 1;
#ifdef _WIN32
    in->c_numeric = _create_locale(LC_NUMERIC, "C");
    if (!in->c_numeric) return 0;
#else
    in->c_numeric = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    if (in->c_numeric == (locale_t)0) return 0;
#endif
    in->c_numeric_ready = 1;
    return 1;
}

/* snprintf in the C locale's LC_NUMERIC ("%.17g" writes '.' whatever the
 * importer's locale).  Falls back to the process locale only if no C
 * locale could be made, which instantiation has already refused. */
static int c_snprintf(Instance *in, char *buf, size_t cap, const char *fmt, ...) {
    va_list ap;
    int n;
    va_start(ap, fmt);
    if (!c_numeric_ready(in)) {
        n = vsnprintf(buf, cap, fmt, ap);
    } else {
#ifdef _WIN32
        n = _vsnprintf_l(buf, cap, fmt, in->c_numeric, ap);
        if (cap > 0) buf[cap - 1] = '\0';
#else
        locale_t prev = uselocale(in->c_numeric);
        n = vsnprintf(buf, cap, fmt, ap);
        uselocale(prev);
#endif
    }
    va_end(ap);
    return n;
}

/* strtod in the C locale's LC_NUMERIC ('.' is the decimal point). */
static double c_strtod(Instance *in, const char *p, char **end) {
    if (!c_numeric_ready(in)) return strtod(p, end);
#ifdef _WIN32
    return _strtod_l(p, end, in->c_numeric);
#else
    locale_t prev = uselocale(in->c_numeric);
    double v = strtod(p, end);
    uselocale(prev);
    return v;
#endif
}

/* Everything an instance holds, the instance included.  The socket is
 * closed if it is still open; nothing is sent. */
static void instance_release(Instance *in) {
    if (!in) return;
    if (in->sock != SOCK_INVALID) sock_close(in->sock);
    free(in->req); free(in->resp);
    if (in->c_numeric_ready) {
#ifdef _WIN32
        _free_locale(in->c_numeric);
#else
        freelocale(in->c_numeric);
#endif
    }
    free(in);
}

/* ------------------------------------------------------------------ util */

static void inst_log(Instance *in, fmi3Status status, const char *category,
                     const char *msg) {
    if (in && in->log) {
        in->log(in->env, status, category, msg);
    }
}

static int phase_allows(Instance *in, unsigned allowed, const char *fn, const char *where) {
    if (in->phase >= PHASE_INSTANTIATED && in->phase <= PHASE_TERMINATED
            && (allowed & (1u << in->phase)))
        return 1;
    char msg[384];
    snprintf(msg, sizeof msg, "maddening_fmu: %s is not allowed in the %s state; FMI 3.0 "
             "allows it %s.  Nothing was sent", fn, phase_name(in->phase), where);
    inst_log(in, fmi3Error, "logStatusError", msg);
    return 0;
}

/* A sidecar that went away must surface as fmi3Error from the next call,
 * never as SIGPIPE killing the importer's process. */
#ifdef MSG_NOSIGNAL
#  define SEND_FLAGS MSG_NOSIGNAL
#else
#  define SEND_FLAGS 0
#endif

/* A signal that interrupts send() before anything is written is not a
 * failure (EINTR): retry it.  Every other failure is final, and the
 * caller drops the connection (bridge_xfer), because part of a frame may
 * already be on the wire and the next frame would be read as its rest. */
static int send_all(sock_t s, const char *buf, size_t n) {
    while (n > 0) {
        ssize_t k = send(s, buf, n, SEND_FLAGS);
#ifdef EINTR
        if (k < 0 && errno == EINTR) continue;
#endif
        if (k <= 0) return -1;
        buf += k; n -= (size_t)k;
    }
    return 0;
}

/* EINTR is retried, as in send_all.  EOF clears errno, so a caller can
 * tell a socket timeout (EAGAIN / EWOULDBLOCK, from SO_RCVTIMEO) from a
 * peer that hung up. */
static int recv_all(sock_t s, char *buf, size_t n) {
    while (n > 0) {
        ssize_t k = recv(s, buf, n, 0);
#ifdef EINTR
        if (k < 0 && errno == EINTR) continue;
#endif
        if (k == 0) {
            errno = 0;
#ifdef _WIN32
            WSASetLastError(0);
#endif
            return -1;
        }
        if (k < 0) return -1;
        buf += k; n -= (size_t)k;
    }
    return 0;
}

/* Did the last failed recv / send time out (SO_RCVTIMEO / SO_SNDTIMEO)? */
static int timed_out(void) {
#ifdef _WIN32
    return WSAGetLastError() == WSAETIMEDOUT;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

/* Receive and send deadlines on `s`, in seconds; 0 removes them. */
static int set_socket_timeout(sock_t s, double seconds) {
#ifdef _WIN32
    DWORD ms = (DWORD)(seconds * 1000.0 + 0.5);
    if (seconds > 0 && ms == 0) ms = 1;
    if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof ms)) return -1;
    if (setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof ms)) return -1;
#else
    struct timeval tv;
    tv.tv_sec = (time_t)seconds;
    tv.tv_usec = (long)((seconds - (double)tv.tv_sec) * 1e6);
    if (seconds > 0 && tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1;
    if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv)) return -1;
    if (setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv)) return -1;
#endif
    return 0;
}

/* MADDENING_FMU_TIMEOUT in seconds (REPLY_TIMEOUT_DEFAULT_S when unset or
 * empty); -1 when it is not a plain number from 0 to REPLY_TIMEOUT_MAX_S.
 * Read in the C locale: "2.5" means two and a half seconds everywhere. */
static int read_timeout(Instance *in, double *out) {
    const char *spec = getenv("MADDENING_FMU_TIMEOUT");
    if (!spec || !*spec) { *out = REPLY_TIMEOUT_DEFAULT_S; return 0; }
    char *end;
    errno = 0;
    double v = c_strtod(in, spec, &end);
    while (*end == ' ') ++end;
    if (end == spec || *end != '\0' || errno == ERANGE || !(v >= 0.0) || v > REPLY_TIMEOUT_MAX_S)
        return -1;
    *out = v;
    return 0;
}

/* Wire order for binary frames is little-endian float64; the host
 * converts only if it is big-endian (a plain memcpy everywhere else). */
static int host_is_little_endian(void) {
    unsigned x = 1u;
    return *(const unsigned char *)&x == 1u;
}

static void f64_from_le(double *out, const unsigned char *src, size_t n) {
    memcpy(out, src, n * sizeof(double));
    if (!host_is_little_endian()) {
        unsigned char *p = (unsigned char *)out;
        for (size_t i = 0; i < n; ++i, p += 8)
            for (size_t j = 0; j < 4; ++j) { unsigned char t = p[j]; p[j] = p[7 - j]; p[7 - j] = t; }
    }
}

static void f64_to_le(unsigned char *dst, const double *src, size_t n) {
    memcpy(dst, src, n * sizeof(double));
    if (!host_is_little_endian()) {
        for (size_t i = 0; i < n; ++i, dst += 8)
            for (size_t j = 0; j < 4; ++j) { unsigned char t = dst[j]; dst[j] = dst[7 - j]; dst[7 - j] = t; }
    }
}

static void put_be32(unsigned char *p, unsigned long v) {
    p[0] = (unsigned char)(v >> 24); p[1] = (unsigned char)(v >> 16);
    p[2] = (unsigned char)(v >> 8);  p[3] = (unsigned char)v;
}

static unsigned long get_be32(const unsigned char *p) {
    return ((unsigned long)p[0] << 24) | ((unsigned long)p[1] << 16)
         | ((unsigned long)p[2] << 8) | (unsigned long)p[3];
}

/* The stream can no longer be trusted (a reply we refused to read, or a
 * peer that stopped mid-frame): close it so that every later call fails
 * honestly instead of parsing whatever bytes come next as its reply. */
static void conn_drop(Instance *in, const char *why) {
    inst_log(in, fmi3Error, "logStatusError", why);
    if (in->sock != SOCK_INVALID) sock_close(in->sock);
    in->sock = SOCK_INVALID;
    FUZZ_COUNT(conn_dropped);
}

/* Send one frame (`n` bytes of `req`; `binary` sets bit 31 of the
 * prefix) and store the reply in in->resp.  A JSON reply must carry
 * "ok":true.  A binary reply is split: its JSON header is copied,
 * NUL-terminated, into in->hdr (which must carry "ok":true) and
 * in->raw / in->raw_len point at the raw part inside in->resp.  Returns
 * fmi3OK or fmi3Error (with the server's error text logged).  A reply
 * longer than FRAME_MAX (either kind) or cut short by the peer drops the
 * connection: the instance is dead from then on (every call fmi3Error),
 * never out of step. */
#define TIMEOUT_MESSAGE "maddening_fmu: the sidecar did not answer within the deadline " \
    "(MADDENING_FMU_TIMEOUT); the connection is closed"

static fmi3Status bridge_xfer(Instance *in, const char *req, size_t n, int binary) {
    unsigned char head[4];
    in->resp_binary = 0; in->raw = NULL; in->raw_len = 0; in->hdr[0] = '\0';
    if (in->resp) in->resp[0] = '\0';
    if (in->sock == SOCK_INVALID) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: connection to the sidecar is closed");
        return fmi3Error;
    }
    if (n > FRAME_LEN_MASK) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: request exceeds the frame size");
        return fmi3Error;
    }
    put_be32(head, (unsigned long)n | (binary ? FRAME_BINARY : 0ul));
    if (send_all(in->sock, (const char *)head, 4) || send_all(in->sock, req, n)) {
        /* Part of the frame may be on the wire: the bridge would read the
         * next request as the rest of this one.  The stream is out of
         * step, so it is closed like a failed receive. */
        conn_drop(in, timed_out() ? TIMEOUT_MESSAGE : "maddening_fmu: send failed");
        return fmi3Error;
    }
    if (recv_all(in->sock, (char *)head, 4)) {
        conn_drop(in, timed_out() ? TIMEOUT_MESSAGE : "maddening_fmu: recv failed");
        return fmi3Error;
    }
    unsigned long word = get_be32(head);
    int reply_binary = (word & FRAME_BINARY) != 0;
    size_t m = (size_t)(word & FRAME_LEN_MASK);
    if (m > FRAME_MAX) {
        /* the peer is out of step with us; never allocate for it, and
         * never read what follows as if it were the next reply */
        conn_drop(in, "maddening_fmu: reply exceeds the frame limit");
        return fmi3Error;
    }
    if (m + 1 > in->resp_cap) {
        char *p = (char *)realloc(in->resp, m + 1);
        if (!p) return fmi3Fatal;
        in->resp = p; in->resp_cap = m + 1;
    }
    /* Invariant: in->resp is always a NUL-terminated string, also after a
     * failed or partial receive (a later parse must never run off the
     * end of a half-filled buffer). */
    in->resp[0] = '\0';
    if (m > 0 && recv_all(in->sock, in->resp, m)) {
        in->resp[0] = '\0';
        conn_drop(in, timed_out() ? TIMEOUT_MESSAGE : "maddening_fmu: recv body failed");
        return fmi3Error;
    }
    in->resp[m] = '\0';
    FUZZ_COUNT(replies_read);
    if (!reply_binary) {
        if (strstr(in->resp, "\"ok\":true") == NULL) {
            const char *e = strstr(in->resp, "\"error\":\"");
            inst_log(in, fmi3Error, "logStatusError", e ? e + 9 : in->resp);
            return fmi3Error;
        }
        return fmi3OK;
    }
    /* binary: [u32 BE header_len][header JSON][raw]; nothing is trusted */
    if (m < 4) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: binary reply too short");
        return fmi3Error;
    }
    unsigned long hl = get_be32((const unsigned char *)in->resp);
    if (hl > m - 4 || hl >= sizeof in->hdr) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: binary reply header is malformed");
        return fmi3Error;
    }
    memcpy(in->hdr, in->resp + 4, hl);
    in->hdr[hl] = '\0';
    in->raw = in->resp + 4 + hl;
    in->raw_len = m - 4 - hl;
    in->resp_binary = 1;
    FUZZ_COUNT(binary_replies);
    if (strstr(in->hdr, "\"ok\":true") == NULL) {
        const char *e = strstr(in->hdr, "\"error\":\"");
        inst_log(in, fmi3Error, "logStatusError", e ? e + 9 : in->hdr);
        return fmi3Error;
    }
    return fmi3OK;
}

/* Send `req` (JSON text) as a JSON frame. */
static fmi3Status bridge_call(Instance *in, const char *req) {
    return bridge_xfer(in, req, strlen(req), 0);
}

/* "n":<count> of the last binary reply's header; -1 if absent or not a
 * plain non-negative decimal followed by the end of the JSON value. */
static int hdr_count(const Instance *in, size_t *count) {
    const char *p = strstr(in->hdr, "\"n\":");
    if (!p) return -1;
    p += 4;
    while (*p == ' ') ++p;
    if (*p < '0' || *p > '9') return -1;
    char *end;
    errno = 0;
    unsigned long long v = strtoull(p, &end, 10);
    if (end == p || errno == ERANGE || v > (unsigned long long)FRAME_MAX) return -1;
    while (*end == ' ') ++end;
    if (*end != ',' && *end != '}') return -1;   /* "1e3", "5abc", "0x10" are not counts */
    *count = (size_t)v;
    return 0;
}

/* The raw doubles of a binary get reply: the header's count must match
 * the raw length exactly and equal the caller's array size (FMI's
 * nValues is the number of values the value references hold, so a reply
 * of any other length answers a different question). */
static fmi3Status parse_binary_values(Instance *in, double *out, size_t n) {
    size_t have;
    FUZZ_COUNT(parse_binary_values);
    if (hdr_count(in, &have)) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: binary reply has no valid count");
        return fmi3Error;
    }
    if (strstr(in->hdr, "\"dtype\":\"f64\"") == NULL) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: binary reply is not float64");
        return fmi3Error;
    }
    if (have > in->raw_len / sizeof(double) || have * sizeof(double) != in->raw_len) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: binary reply length mismatch");
        return fmi3Error;
    }
    if (have < n) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: too few values in reply");
        return fmi3Error;
    }
    if (have > n) {
        /* nValues smaller than what the value references hold: answering
         * fmi3OK with the first nValues dropped the rest in silence */
        inst_log(in, fmi3Error, "logStatusError",
                 "maddening_fmu: more values in reply than nValues (nValues must be the "
                 "number of values the value references hold)");
        return fmi3Error;
    }
    if (n > 0) { f64_from_le(out, (const unsigned char *)in->raw, n); FUZZ_COUNT(parse_binary_ok); }
    return fmi3OK;
}

/* Parse "values":[n0,n1,...] from in->resp into out[0..n): exactly n
 * values, as in the binary path. */
static fmi3Status parse_values(Instance *in, double *out, size_t n) {
    FUZZ_COUNT(parse_values);
    if (n == 0) return fmi3OK;
    if (in->resp == NULL) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: no reply to parse");
        return fmi3Error;
    }
    const char *p = strstr(in->resp, "\"values\":[");
    if (!p) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: reply has no values");
        return fmi3Error;
    }
    p += 10;
    for (size_t i = 0; i < n; ++i) {
        char *end;
        while (*p == ' ' || *p == ',') ++p;
        if (*p == ']' || *p == '\0') {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: too few values in reply");
            return fmi3Error;
        }
        /* A non-finite value arrives as a *quoted* token -- "NaN",
         * "Infinity", "-Infinity" -- because the bare tokens json.dumps
         * used to write are not JSON and no conforming parser but
         * Python's accepts them (MADD-ANO-006).  C99 strtod parses the
         * token itself, case-insensitively, including "infinity"; the
         * quote is the only thing it cannot step over, so step over it
         * here.  The bare form is still accepted, so this wrapper reads
         * a reply from a bridge of either vintage. */
        int quoted = (*p == '"');
        if (quoted) ++p;
        out[i] = c_strtod(in, p, &end);
        if (end == p) {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: malformed number in reply");
            return fmi3Error;
        }
        p = end;
        if (quoted) {
            /* Only a closing quote may follow: "1e5xyz" must not pass as
             * 1e5 with trailing junk silently dropped. */
            if (*p != '"') {
                inst_log(in, fmi3Error, "logStatusError",
                         "maddening_fmu: malformed quoted number in reply");
                return fmi3Error;
            }
            ++p;
        }
    }
    while (*p == ' ') ++p;
    if (*p != ']') {
        inst_log(in, fmi3Error, "logStatusError",
                 *p == ',' ? "maddening_fmu: more values in reply than nValues (nValues must be "
                             "the number of values the value references hold)"
                           : "maddening_fmu: malformed values list in reply");
        return fmi3Error;
    }
    FUZZ_COUNT(parse_values_ok);
    return fmi3OK;
}

static int req_reserve(Instance *in, size_t need) {
    if (need <= in->req_cap) return 0;
    size_t cap = in->req_cap ? in->req_cap : 4096;
    while (cap < need) cap *= 2;
    char *p = (char *)realloc(in->req, cap);
    if (!p) return -1;
    in->req = p; in->req_cap = cap;
    return 0;
}

/* ,"type":"<FMI type>" for a request, or nothing when type is NULL.  The
 * type is a literal from the DEFINE_GET / DEFINE_SET table below (no JSON
 * escaping needed); the bridge checks it against each variable's type. */
static int put_type(char *w, const char *type) {
    return type ? sprintf(w, ",\"type\":\"%s\"", type) : 0;
}

/* Build a set request from any numeric width: a binary frame
 * ({"op":"set","type":T,"vr":[..],"n":N,"dtype":"f64"} + N raw
 * little-endian doubles) after a protocol-2 hello, else the JSON
 * {"op":"set","type":T,"vr":[..],"values":[..]} with %.17g text.  T is
 * the FMI type of the setter that was called. */
static fmi3Status do_set(Instance *in, const char *type, const fmi3ValueReference vr[],
                         size_t nvr, const double values[], size_t nvalues) {
    static const char *const too_big = "maddening_fmu: set exceeds the frame limit";
    /* Cheap bounds before any buffer grows: 8 bytes per value and at
     * least 2 header bytes per value reference can never fit the frame. */
    if (nvalues > FRAME_MAX / sizeof(double) || nvr > FRAME_MAX / 2) {
        inst_log(in, fmi3Error, "logStatusError", too_big);
        return fmi3Error;
    }
    for (size_t i = 0; i < nvalues; ++i) {
        if (isnan(values[i]) || isinf(values[i])) {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: non-finite value");
            return fmi3Error;
        }
    }
    if (in->binary) {
        /* header first, so the exact frame length is known before the
         * raw part is laid out: [u32 hl][header][8 * nvalues] <= FRAME_MAX */
        if (req_reserve(in, 4 + 128 + 24 * nvr)) return fmi3Fatal;
        char *h = in->req + 4;
        char *w = h;
        w += sprintf(w, "{\"op\":\"set\"");
        w += put_type(w, type);
        w += sprintf(w, ",\"vr\":[");
        for (size_t i = 0; i < nvr; ++i) w += sprintf(w, "%s%u", i ? "," : "", (unsigned)vr[i]);
        w += sprintf(w, "],\"n\":%lu,\"dtype\":\"f64\"}", (unsigned long)nvalues);
        size_t hl = (size_t)(w - h);
        size_t total = 4 + hl + sizeof(double) * nvalues;
        if (total > FRAME_MAX) {
            inst_log(in, fmi3Error, "logStatusError", too_big);
            return fmi3Error;
        }
        if (req_reserve(in, total)) return fmi3Fatal;       /* realloc keeps the header */
        put_be32((unsigned char *)in->req, (unsigned long)hl);
        f64_to_le((unsigned char *)in->req + 4 + hl, values, nvalues);
        return bridge_xfer(in, in->req, total, 1);
    }
    if (req_reserve(in, 96 + 24 * nvr + 32 * nvalues)) return fmi3Fatal;
    char *w = in->req;
    w += sprintf(w, "{\"op\":\"set\"");
    w += put_type(w, type);
    w += sprintf(w, ",\"vr\":[");
    for (size_t i = 0; i < nvr; ++i) w += sprintf(w, "%s%u", i ? "," : "", (unsigned)vr[i]);
    w += sprintf(w, "],\"values\":[");
    for (size_t i = 0; i < nvalues; ++i)
        w += c_snprintf(in, w, in->req_cap - (size_t)(w - in->req), "%s%.17g", i ? "," : "",
                        values[i]);
    w += sprintf(w, "]}");
    if ((size_t)(w - in->req) > FRAME_MAX) {
        inst_log(in, fmi3Error, "logStatusError", too_big);
        return fmi3Error;
    }
    return bridge_call(in, in->req);
}

static fmi3Status do_get(Instance *in, const char *type, const fmi3ValueReference vr[],
                         size_t nvr, double *out, size_t nvalues) {
    /* The same frame check do_set has.  Without it a get of a few million
     * value references built a request frame over FRAME_MAX, which the
     * bridge refuses to read: the connection is dropped and the instance
     * dies, instead of the importer being told its request is too large. */
    static const char *const too_big = "maddening_fmu: get exceeds the frame limit";
    if (nvr > FRAME_MAX / 2) {
        inst_log(in, fmi3Error, "logStatusError", too_big);
        return fmi3Error;
    }
    if (req_reserve(in, 96 + 24 * nvr)) return fmi3Fatal;
    char *w = in->req;
    w += sprintf(w, "{\"op\":\"get\"");
    w += put_type(w, type);
    w += sprintf(w, ",\"vr\":[");
    for (size_t i = 0; i < nvr; ++i) w += sprintf(w, "%s%u", i ? "," : "", (unsigned)vr[i]);
    w += sprintf(w, "]}");
    if ((size_t)(w - in->req) > FRAME_MAX) {
        inst_log(in, fmi3Error, "logStatusError", too_big);
        return fmi3Error;
    }
    fmi3Status st = bridge_call(in, in->req);
    if (st != fmi3OK) return st;
    if (in->resp_binary) return parse_binary_values(in, out, nvalues);
    return parse_values(in, out, nvalues);
}

/* A hello reply negotiates binary frames when it is a JSON frame that
 * carries "protocol" >= 2 and "binary":true; anything else (an older
 * bridge, in particular) leaves the instance on JSON. */
static int hello_negotiated_binary(const Instance *in) {
    FUZZ_COUNT(hello_negotiated);
    if (in->resp == NULL || in->resp_binary) return 0;
    const char *p = strstr(in->resp, "\"protocol\":");
    if (!p) return 0;
    long v = strtol(p + 11, NULL, 10);
    if (v < 2 || strstr(in->resp, "\"binary\":true") == NULL) return 0;
    FUZZ_COUNT(hello_binary);
    return 1;
}

/* Does the hello reply's "token" equal the importer's instantiationToken,
 * exactly?  FMI requires the importer to pass the modelDescription's token;
 * a NULL or empty one is a mismatch (it used to skip the check), and the
 * comparison is of the whole string, so neither may be a prefix of the
 * other. */
static int token_matches(const Instance *in, fmi3String token) {
    if (token == NULL || *token == '\0' || in->resp == NULL || in->resp_binary) return 0;
    const char *p = strstr(in->resp, "\"token\":\"");
    if (p == NULL) return 0;
    p += 9;
    const char *q = strchr(p, '"');
    if (q == NULL) return 0;
    size_t n = strlen(token);
    return (size_t)(q - p) == n && strncmp(p, token, n) == 0;
}

static int read_endpoint(fmi3String resource_path, char *host, size_t hostcap, int *port) {
    const char *spec = getenv("MADDENING_FMU_ENDPOINT");
    char buf[512] = {0};
    if (!spec && resource_path && *resource_path) {
        char path[1024];
        const char *rp = resource_path;
        if (strncmp(rp, "file://", 7) == 0) rp += 7;
        size_t rl = strlen(rp);
        if (rl > 0) {
            snprintf(path, sizeof path, "%s%sendpoint.txt", rp,
                     (rp[rl - 1] == '/' || rp[rl - 1] == '\\') ? "" : "/");
            FILE *f = fopen(path, "r");
            if (f) {
                if (fgets(buf, sizeof buf, f)) {
                    buf[strcspn(buf, "\r\n")] = '\0';     /* trailing newline */
                    spec = buf;
                }
                fclose(f);
            }
        }
    }
    if (!spec) return -1;
    const char *colon = strrchr(spec, ':');
    if (!colon) return -1;
    size_t hl = (size_t)(colon - spec);
    /* "[::1]:5555": strip the brackets around an IPv6 literal */
    if (hl >= 2 && spec[0] == '[' && spec[hl - 1] == ']') { spec += 1; hl -= 2; }
    if (hl == 0 || hl + 1 > hostcap) return -1;
    memcpy(host, spec, hl); host[hl] = '\0';
    *port = atoi(colon + 1);
    return *port > 0 ? 0 : -1;
}

/* `timeout` (seconds, 0 = none) is set on each socket before connect:
 * on Linux SO_SNDTIMEO bounds connect() itself, and on every platform it
 * bounds the hello exchange that follows. */
static sock_t connect_endpoint(const char *host, int port, double timeout) {
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    struct addrinfo hints, *res = NULL;
    char portstr[16];
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    snprintf(portstr, sizeof portstr, "%d", port);
    if (getaddrinfo(host, portstr, &hints, &res) != 0) return SOCK_INVALID;
    sock_t s = SOCK_INVALID;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == SOCK_INVALID) continue;
#ifdef SO_NOSIGPIPE
        { int one = 1; setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one); }
#endif
        /* Every call is a small request answered before the next is sent,
         * and a frame goes out as two writes (prefix, then body).  With
         * Nagle's algorithm on, the body waited for the ACK of the prefix,
         * which the bridge's kernel delays (40 ms on Linux): every FMI call
         * took at least that long.  Measured: 245 calls of an 80-step FMPy
         * run took 10.1 s, about 41 ms each. */
        { int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one); }
        if (set_socket_timeout(s, timeout) == 0
            && connect(s, ai->ai_addr, (int)ai->ai_addrlen) == 0) break;
        sock_close(s); s = SOCK_INVALID;
    }
    freeaddrinfo(res);
    return s;
}

/* ----------------------------------------------------- FMI 3.0 exports */

FMI3_Export const char *fmi3GetVersion(void) { return fmi3Version; }

FMI3_Export fmi3Status fmi3SetDebugLogging(fmi3Instance instance, fmi3Boolean loggingOn,
                                           size_t nCategories, const fmi3String categories[]) {
    (void)nCategories; (void)categories;
    Instance *in = (Instance *)instance;
    if (in) in->logging_on = loggingOn;
    return fmi3OK;
}

FMI3_Export fmi3Instance fmi3InstantiateModelExchange(
    fmi3String instanceName, fmi3String instantiationToken, fmi3String resourcePath,
    fmi3Boolean visible, fmi3Boolean loggingOn, fmi3InstanceEnvironment instanceEnvironment,
    fmi3LogMessageCallback logMessage) {
    (void)instanceName; (void)instantiationToken; (void)resourcePath; (void)visible;
    (void)loggingOn; (void)instanceEnvironment;
    if (logMessage) logMessage(instanceEnvironment, fmi3Error, "logStatusError",
                               "maddening_fmu: model exchange is not supported");
    return NULL;
}

FMI3_Export fmi3Instance fmi3InstantiateCoSimulation(
    fmi3String instanceName, fmi3String instantiationToken, fmi3String resourcePath,
    fmi3Boolean visible, fmi3Boolean loggingOn, fmi3Boolean eventModeUsed,
    fmi3Boolean earlyReturnAllowed, const fmi3ValueReference requiredIntermediateVariables[],
    size_t nRequiredIntermediateVariables, fmi3InstanceEnvironment instanceEnvironment,
    fmi3LogMessageCallback logMessage, fmi3IntermediateUpdateCallback intermediateUpdate) {
    (void)visible; (void)eventModeUsed; (void)earlyReturnAllowed;
    (void)requiredIntermediateVariables; (void)nRequiredIntermediateVariables;
    (void)intermediateUpdate;
    Instance *in = (Instance *)calloc(1, sizeof *in);
    if (!in) return NULL;
    in->sock = SOCK_INVALID;
    in->env = instanceEnvironment; in->log = logMessage; in->logging_on = loggingOn;
    snprintf(in->instance_name, sizeof in->instance_name, "%s", instanceName ? instanceName : "");
    if (!c_numeric_ready(in)) {
        inst_log(in, fmi3Error, "logStatusError",
                 "maddening_fmu: cannot make a C LC_NUMERIC locale to read and write numbers in");
        instance_release(in); return NULL;
    }

    char host[256]; int port = 0;
    if (read_endpoint(resourcePath, host, sizeof host, &port)) {
        inst_log(in, fmi3Error, "logStatusError",
                 "maddening_fmu: no endpoint (resources/endpoint.txt or MADDENING_FMU_ENDPOINT)");
        instance_release(in); return NULL;
    }
    if (read_timeout(in, &in->timeout_s)) {
        inst_log(in, fmi3Error, "logStatusError",
                 "maddening_fmu: MADDENING_FMU_TIMEOUT must be a number of seconds from 0 "
                 "(no deadline) to 1e6");
        instance_release(in); return NULL;
    }
    /* the hello is answered at once by a live bridge: a short deadline */
    double handshake = in->timeout_s > 0 && in->timeout_s < HANDSHAKE_TIMEOUT_S
                       ? in->timeout_s : (in->timeout_s > 0 ? HANDSHAKE_TIMEOUT_S : 0.0);
    in->sock = connect_endpoint(host, port, handshake);
    if (in->sock == SOCK_INVALID) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: cannot connect to sidecar");
        instance_release(in); return NULL;
    }
    if (bridge_call(in, "{\"op\":\"hello\",\"protocol\":2,\"binary\":true}") != fmi3OK) {
        instance_release(in); return NULL;
    }
    in->binary = hello_negotiated_binary(in);
    if (!token_matches(in, instantiationToken)) {
        inst_log(in, fmi3Error, "logStatusError",
                 "maddening_fmu: instantiation token does not match the sidecar's graph");
        instance_release(in); return NULL;
    }
    /* from here every call may be a long step: the full deadline */
    if (set_socket_timeout(in->sock, in->timeout_s)) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: cannot set the socket deadline");
        instance_release(in); return NULL;
    }
    return (fmi3Instance)in;
}

FMI3_Export fmi3Instance fmi3InstantiateScheduledExecution(
    fmi3String instanceName, fmi3String instantiationToken, fmi3String resourcePath,
    fmi3Boolean visible, fmi3Boolean loggingOn, fmi3InstanceEnvironment instanceEnvironment,
    fmi3LogMessageCallback logMessage, fmi3ClockUpdateCallback clockUpdate,
    fmi3LockPreemptionCallback lockPreemption, fmi3UnlockPreemptionCallback unlockPreemption) {
    (void)instanceName; (void)instantiationToken; (void)resourcePath; (void)visible;
    (void)loggingOn; (void)instanceEnvironment; (void)clockUpdate; (void)lockPreemption;
    (void)unlockPreemption;
    if (logMessage) logMessage(instanceEnvironment, fmi3Error, "logStatusError",
                               "maddening_fmu: scheduled execution is not supported");
    return NULL;
}

FMI3_Export void fmi3FreeInstance(fmi3Instance instance) {
    Instance *in = (Instance *)instance;
    if (!in) return;
    if (in->sock != SOCK_INVALID) bridge_call(in, "{\"op\":\"terminate\"}");
    instance_release(in);
}

/* The start time is the instance's time from here on, so the bridge is
 * told it ("initialize"): it used to stay in in->time, and the "time"
 * variable read 0.0 until the first step.  The bridge then requires the
 * first doStep at this time. */
FMI3_Export fmi3Status fmi3EnterInitializationMode(
    fmi3Instance instance, fmi3Boolean toleranceDefined, fmi3Float64 tolerance,
    fmi3Float64 startTime, fmi3Boolean stopTimeDefined, fmi3Float64 stopTime) {
    (void)toleranceDefined; (void)tolerance; (void)stopTimeDefined; (void)stopTime;
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    if (!phase_allows(in, IN_INSTANTIATED, "fmi3EnterInitializationMode",
                      "in the Instantiated state only (after instantiation or fmi3Reset)"))
        return fmi3Error;
    if (!isfinite(startTime)) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: start time must be finite");
        return fmi3Error;
    }
    char req[64];
    c_snprintf(in, req, sizeof req, "{\"op\":\"initialize\",\"t\":%.17g}", startTime);
    fmi3Status st = bridge_call(in, req);
    if (st != fmi3OK) return st;
    in->time = startTime;
    in->phase = PHASE_INITIALIZATION_MODE;
    return fmi3OK;
}

/* Without Event Mode, leaving Initialization Mode enters Step Mode. */
FMI3_Export fmi3Status fmi3ExitInitializationMode(fmi3Instance instance) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    if (!phase_allows(in, IN_INITIALIZATION, "fmi3ExitInitializationMode",
                      "in Initialization Mode only"))
        return fmi3Error;
    in->phase = PHASE_STEP_MODE;
    return fmi3OK;
}

FMI3_Export fmi3Status fmi3EnterEventMode(fmi3Instance instance) {
    /* modelDescription advertises hasEventMode="false" */
    (void)instance; return fmi3Error;
}

FMI3_Export fmi3Status fmi3Terminate(fmi3Instance instance) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    if (!phase_allows(in, IN_STEP_MODE, "fmi3Terminate", "in Step Mode only"))
        return fmi3Error;
    fmi3Status st = bridge_call(in, "{\"op\":\"terminate\"}");
    if (st == fmi3OK) in->phase = PHASE_TERMINATED;
    return st;
}

/* The wrapper's clock goes back to zero, and the instance to Instantiated,
 * only once the bridge has reset: a failed reset leaves the instance, its
 * state and its time where they were. */
FMI3_Export fmi3Status fmi3Reset(fmi3Instance instance) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    fmi3Status st = bridge_call(in, "{\"op\":\"reset\"}");
    if (st == fmi3OK) { in->time = 0.0; in->phase = PHASE_INSTANTIATED; }
    return st;
}

/* ---- getters and setters: every numeric width goes through double ----
 *
 * The request names the FMI type of the function called, and the bridge
 * refuses a variable of another type.  A getter then checks each value
 * before converting it: a double that its C type cannot hold -- NaN, a
 * fraction or an out-of-range number for an integer type, anything but 0
 * or 1 for a Boolean, a finite number beyond FLT_MAX for a Float32 -- is
 * fmi3Error with values[] untouched, because the conversion would be
 * undefined behaviour or a silent change of value (fmi3GetInt32 on a
 * Float32 output used to return 0 for 0.5, with fmi3OK).  The range checks
 * short-circuit before the cast, and every bound is a power of two, exact
 * in a double. */

#define FITS_ANY(v)  1
#define FITS_F32(v)  (!isfinite(v) || ((v) >= -FLT_MAX && (v) <= FLT_MAX))
#define FITS_INT(v, LO, HI, CTYPE) ((v) >= (LO) && (v) < (HI) && (double)(CTYPE)(v) == (v))
#define FITS_I8(v)   FITS_INT(v, -128.0, 128.0, fmi3Int8)
#define FITS_U8(v)   FITS_INT(v, 0.0, 256.0, fmi3UInt8)
#define FITS_I16(v)  FITS_INT(v, -32768.0, 32768.0, fmi3Int16)
#define FITS_U16(v)  FITS_INT(v, 0.0, 65536.0, fmi3UInt16)
#define FITS_I32(v)  FITS_INT(v, -2147483648.0, 2147483648.0, fmi3Int32)
#define FITS_U32(v)  FITS_INT(v, 0.0, 4294967296.0, fmi3UInt32)
#define FITS_I64(v)  FITS_INT(v, -9223372036854775808.0, 9223372036854775808.0, fmi3Int64)
#define FITS_U64(v)  FITS_INT(v, 0.0, 18446744073709551616.0, fmi3UInt64)
#define FITS_BOOL(v) ((v) == 0.0 || (v) == 1.0)

/* An Int64 / UInt64 value the double on the wire cannot carry exactly
 * (|v| > 2^53, most of them) is refused rather than rounded; every other
 * width converts to double exactly. */
#define EXACT_ANY(v) 1
#define EXACT_I64(v) ((double)(v) < 9223372036854775808.0 && (fmi3Int64)(double)(v) == (v))
#define EXACT_U64(v) ((double)(v) < 18446744073709551616.0 && (fmi3UInt64)(double)(v) == (v))

static fmi3Status refuse_reply_value(Instance *in, const char *fn, const char *type, double v) {
    char msg[256];
    snprintf(msg, sizeof msg, "maddening_fmu: %s: the sidecar sent %.17g, which is not a value "
             "of type %s; nothing was written to values[]", fn, v, type);
    inst_log(in, fmi3Error, "logStatusError", msg);
    return fmi3Error;
}

static fmi3Status refuse_inexact_value(Instance *in, const char *fn, size_t i) {
    char msg[256];
    snprintf(msg, sizeof msg, "maddening_fmu: %s: values[%lu] cannot be carried exactly as the "
             "float64 the sidecar protocol uses (its magnitude is above 2^53); nothing was sent",
             fn, (unsigned long)i);
    inst_log(in, fmi3Error, "logStatusError", msg);
    return fmi3Error;
}

#define DEFINE_GET(NAME, CTYPE, TYPE, FITS)                                            \
FMI3_Export fmi3Status NAME(fmi3Instance instance, const fmi3ValueReference vr[],       \
                            size_t nvr, CTYPE values[], size_t nValues) {              \
    Instance *in = (Instance *)instance;                                               \
    if (!in) return fmi3Error;                                                         \
    if (!phase_allows(in, ALLOWED_GET, #NAME,                                          \
                      "in Initialization Mode, Step Mode and Terminated"))             \
        return fmi3Error;                                                              \
    if (nValues == 0) return fmi3OK;                                                   \
    double *tmp = (double *)malloc(nValues * sizeof(double));                          \
    if (!tmp) return fmi3Fatal;                                                        \
    fmi3Status st = do_get(in, TYPE, vr, nvr, tmp, nValues);                           \
    for (size_t i = 0; st == fmi3OK && i < nValues; ++i)                               \
        if (!(FITS(tmp[i])))                                                           \
            st = refuse_reply_value(in, #NAME, TYPE, tmp[i]);                          \
    if (st == fmi3OK) for (size_t i = 0; i < nValues; ++i) values[i] = (CTYPE)tmp[i];  \
    free(tmp);                                                                         \
    return st;                                                                         \
}

#define DEFINE_SET(NAME, CTYPE, TYPE, EXACT)                                           \
FMI3_Export fmi3Status NAME(fmi3Instance instance, const fmi3ValueReference vr[],       \
                            size_t nvr, const CTYPE values[], size_t nValues) {        \
    Instance *in = (Instance *)instance;                                               \
    if (!in) return fmi3Error;                                                         \
    if (!phase_allows(in, ALLOWED_SET, #NAME,                                          \
                      "in Instantiated, Initialization Mode and Step Mode"))           \
        return fmi3Error;                                                              \
    if (nValues == 0) return fmi3OK;                                                   \
    for (size_t i = 0; i < nValues; ++i)                                               \
        if (!(EXACT(values[i]))) return refuse_inexact_value(in, #NAME, i);           \
    double *tmp = (double *)malloc(nValues * sizeof(double));                          \
    if (!tmp) return fmi3Fatal;                                                        \
    for (size_t i = 0; i < nValues; ++i) tmp[i] = (double)values[i];                   \
    fmi3Status st = do_set(in, TYPE, vr, nvr, tmp, nValues);                           \
    free(tmp);                                                                         \
    return st;                                                                         \
}

DEFINE_GET(fmi3GetFloat32, fmi3Float32, "Float32", FITS_F32)
DEFINE_GET(fmi3GetFloat64, fmi3Float64, "Float64", FITS_ANY)
DEFINE_GET(fmi3GetInt8,    fmi3Int8,    "Int8",    FITS_I8)
DEFINE_GET(fmi3GetUInt8,   fmi3UInt8,   "UInt8",   FITS_U8)
DEFINE_GET(fmi3GetInt16,   fmi3Int16,   "Int16",   FITS_I16)
DEFINE_GET(fmi3GetUInt16,  fmi3UInt16,  "UInt16",  FITS_U16)
DEFINE_GET(fmi3GetInt32,   fmi3Int32,   "Int32",   FITS_I32)
DEFINE_GET(fmi3GetUInt32,  fmi3UInt32,  "UInt32",  FITS_U32)
DEFINE_GET(fmi3GetInt64,   fmi3Int64,   "Int64",   FITS_I64)
DEFINE_GET(fmi3GetUInt64,  fmi3UInt64,  "UInt64",  FITS_U64)
DEFINE_GET(fmi3GetBoolean, fmi3Boolean, "Boolean", FITS_BOOL)

DEFINE_SET(fmi3SetFloat32, fmi3Float32, "Float32", EXACT_ANY)
DEFINE_SET(fmi3SetFloat64, fmi3Float64, "Float64", EXACT_ANY)
DEFINE_SET(fmi3SetInt8,    fmi3Int8,    "Int8",    EXACT_ANY)
DEFINE_SET(fmi3SetUInt8,   fmi3UInt8,   "UInt8",   EXACT_ANY)
DEFINE_SET(fmi3SetInt16,   fmi3Int16,   "Int16",   EXACT_ANY)
DEFINE_SET(fmi3SetUInt16,  fmi3UInt16,  "UInt16",  EXACT_ANY)
DEFINE_SET(fmi3SetInt32,   fmi3Int32,   "Int32",   EXACT_ANY)
DEFINE_SET(fmi3SetUInt32,  fmi3UInt32,  "UInt32",  EXACT_ANY)
DEFINE_SET(fmi3SetInt64,   fmi3Int64,   "Int64",   EXACT_I64)
DEFINE_SET(fmi3SetUInt64,  fmi3UInt64,  "UInt64",  EXACT_U64)
DEFINE_SET(fmi3SetBoolean, fmi3Boolean, "Boolean", EXACT_ANY)

FMI3_Export fmi3Status fmi3GetString(fmi3Instance instance, const fmi3ValueReference vr[],
                                     size_t nvr, fmi3String values[], size_t nValues) {
    (void)vr; (void)nvr; (void)values; (void)nValues;
    inst_log((Instance *)instance, fmi3Error, "logStatusError", "maddening_fmu: no String variables");
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3SetString(fmi3Instance instance, const fmi3ValueReference vr[],
                                     size_t nvr, const fmi3String values[], size_t nValues) {
    (void)vr; (void)nvr; (void)values; (void)nValues;
    inst_log((Instance *)instance, fmi3Error, "logStatusError", "maddening_fmu: no String variables");
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetBinary(fmi3Instance instance, const fmi3ValueReference vr[],
                                     size_t nvr, size_t valueSizes[], fmi3Binary values[],
                                     size_t nValues) {
    (void)instance; (void)vr; (void)nvr; (void)valueSizes; (void)values; (void)nValues;
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3SetBinary(fmi3Instance instance, const fmi3ValueReference vr[],
                                     size_t nvr, const size_t valueSizes[],
                                     const fmi3Binary values[], size_t nValues) {
    (void)instance; (void)vr; (void)nvr; (void)valueSizes; (void)values; (void)nValues;
    return fmi3Error;
}
/* FMI 3.0 allows fmi3GetClock and fmi3SetClock only in Event Mode, which
 * this FMU does not have (hasEventMode="false"; fmi3EnterEventMode is
 * refused): its clocks are constant-interval, their ticks implied by
 * time.  Both used to answer fmi3OK for any value reference, clock or
 * not, known or not, and fmi3GetClock reported every one inactive. */
static const char *const no_event_mode =
    "maddening_fmu: fmi3GetClock / fmi3SetClock are allowed only in Event Mode, which this "
    "FMU does not have (hasEventMode=\"false\"); its clocks are constant-interval and tick "
    "with time";
FMI3_Export fmi3Status fmi3GetClock(fmi3Instance instance, const fmi3ValueReference vr[],
                                    size_t nvr, fmi3Clock values[]) {
    (void)vr; (void)nvr; (void)values;
    inst_log((Instance *)instance, fmi3Error, "logStatusError", no_event_mode);
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3SetClock(fmi3Instance instance, const fmi3ValueReference vr[],
                                    size_t nvr, const fmi3Clock values[]) {
    (void)vr; (void)nvr; (void)values;
    inst_log((Instance *)instance, fmi3Error, "logStatusError", no_event_mode);
    return fmi3Error;
}

FMI3_Export fmi3Status fmi3GetNumberOfVariableDependencies(fmi3Instance instance,
                                                           fmi3ValueReference vr,
                                                           size_t *nDependencies) {
    (void)instance; (void)vr; *nDependencies = 0; return fmi3OK;
}
FMI3_Export fmi3Status fmi3GetVariableDependencies(
    fmi3Instance instance, fmi3ValueReference dependent, size_t elementIndicesOfDependent[],
    fmi3ValueReference independents[], size_t elementIndicesOfIndependents[],
    fmi3DependencyKind dependencyKinds[], size_t nDependencies) {
    (void)instance; (void)dependent; (void)elementIndicesOfDependent; (void)independents;
    (void)elementIndicesOfIndependents; (void)dependencyKinds; (void)nDependencies;
    return fmi3OK;
}

/* ---- FMU state: an opaque blob produced by the sidecar (raw npz bytes
 * on the binary protocol, base64 text on JSON) ---- */

typedef struct { char *blob; size_t n; } FmuState;

/* After a successful set_state the bridge's reply carries the restored
 * time ({"ok":true,"t":...}); it becomes the wrapper's clock, which
 * fmi3DoStep reports as lastSuccessfulTime when it fails (it used to
 * report the time before the restore).  A reply without "t" -- an older
 * bridge -- leaves the clock as it was.  Passes `st` through. */
static fmi3Status adopt_restored_time(Instance *in, fmi3Status st) {
    if (st != fmi3OK || in->resp == NULL || in->resp_binary) return st;
    const char *p = strstr(in->resp, "\"t\":");
    if (p == NULL) return st;
    char *end;
    double t = c_strtod(in, p + 4, &end);
    if (end != p + 4 && isfinite(t)) in->time = t;
    return st;
}

FMI3_Export fmi3Status fmi3GetFMUState(fmi3Instance instance, fmi3FMUState *FMUState) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    if (bridge_call(in, "{\"op\":\"get_state\"}") != fmi3OK) return fmi3Error;
    if (in->resp == NULL) return fmi3Error;
    if (in->resp_binary) {
        size_t n;
        FUZZ_COUNT(raw_state);
        if (hdr_count(in, &n) || n != in->raw_len) {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: binary state length mismatch");
            return fmi3Error;
        }
        FmuState *st = (FmuState *)malloc(sizeof *st);
        if (!st) return fmi3Fatal;
        st->blob = (char *)malloc(n + 1);
        if (!st->blob) { free(st); return fmi3Fatal; }
        memcpy(st->blob, in->raw, n); st->blob[n] = '\0'; st->n = n;
        *FMUState = st;
        FUZZ_COUNT(raw_state_ok);
        return fmi3OK;
    }
    const char *p = strstr(in->resp, "\"state\":\"");
    if (!p) return fmi3Error;
    p += 9;
    const char *q = strchr(p, '"');
    if (!q) return fmi3Error;
    FmuState *st = (FmuState *)malloc(sizeof *st);
    if (!st) return fmi3Fatal;
    st->n = (size_t)(q - p);
    st->blob = (char *)malloc(st->n + 1);
    if (!st->blob) { free(st); return fmi3Fatal; }
    memcpy(st->blob, p, st->n); st->blob[st->n] = '\0';
    *FMUState = st;
    return fmi3OK;
}
FMI3_Export fmi3Status fmi3SetFMUState(fmi3Instance instance, fmi3FMUState FMUState) {
    Instance *in = (Instance *)instance;
    FmuState *st = (FmuState *)FMUState;
    if (!in || !st) return fmi3Error;
    static const char *const too_big = "maddening_fmu: FMU state exceeds the frame limit";
    if (in->binary) {
        /* length-delimited raw bytes: nothing the importer supplies can
         * break the framing, the bridge validates the archive itself.
         * The frame is [u32 hl][header][blob], and the bridge drops a
         * connection whose frame exceeds FRAME_MAX, so the check is on the
         * whole frame, as in do_set: a blob of exactly FRAME_MAX bytes used
         * to pass a check on the blob alone and kill the instance. */
        char hdr[64];
        int hl = snprintf(hdr, sizeof hdr, "{\"op\":\"set_state\",\"n\":%lu}", (unsigned long)st->n);
        if (hl <= 0 || (size_t)hl >= sizeof hdr) return fmi3Error;
        if (st->n > FRAME_MAX - 4 - (size_t)hl) {
            inst_log(in, fmi3Error, "logStatusError", too_big);
            return fmi3Error;
        }
        if (req_reserve(in, 4 + (size_t)hl + st->n)) return fmi3Fatal;
        put_be32((unsigned char *)in->req, (unsigned long)hl);
        memcpy(in->req + 4, hdr, (size_t)hl);
        memcpy(in->req + 4 + hl, st->blob, st->n);
        return adopt_restored_time(in, bridge_xfer(in, in->req, 4 + (size_t)hl + st->n, 1));
    }
    /* The same whole-frame limit on the JSON path, checked before the blob
     * is scanned or the request buffer grows. */
    static const char json_frame[] = "{\"op\":\"set_state\",\"state\":\"\"}";
    if (st->n > FRAME_MAX - (sizeof json_frame - 1)) {
        inst_log(in, fmi3Error, "logStatusError", too_big);
        return fmi3Error;
    }
    /* The blob is embedded verbatim in a JSON string: only the base64
     * alphabet the bridge produces is allowed, so importer-supplied bytes
     * (fmi3DeserializeFMUState) can never break the request framing. */
    for (size_t i = 0; i < st->n; ++i) {
        unsigned char c = (unsigned char)st->blob[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
              || c == '+' || c == '/' || c == '=')) {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: FMU state blob is not valid");
            return fmi3Error;
        }
    }
    if (req_reserve(in, st->n + 64)) return fmi3Fatal;
    sprintf(in->req, "{\"op\":\"set_state\",\"state\":\"%s\"}", st->blob);
    return adopt_restored_time(in, bridge_call(in, in->req));
}
FMI3_Export fmi3Status fmi3FreeFMUState(fmi3Instance instance, fmi3FMUState *FMUState) {
    (void)instance;
    FmuState *st = (FmuState *)*FMUState;
    if (st) { free(st->blob); free(st); }
    *FMUState = NULL;
    return fmi3OK;
}
FMI3_Export fmi3Status fmi3SerializedFMUStateSize(fmi3Instance instance, fmi3FMUState FMUState,
                                                  size_t *size) {
    (void)instance;
    FmuState *st = (FmuState *)FMUState;
    if (!st) return fmi3Error;
    *size = st->n;
    return fmi3OK;
}
FMI3_Export fmi3Status fmi3SerializeFMUState(fmi3Instance instance, fmi3FMUState FMUState,
                                             fmi3Byte serializedState[], size_t size) {
    (void)instance;
    FmuState *st = (FmuState *)FMUState;
    if (!st || size < st->n) return fmi3Error;
    memcpy(serializedState, st->blob, st->n);
    return fmi3OK;
}
FMI3_Export fmi3Status fmi3DeserializeFMUState(fmi3Instance instance,
                                               const fmi3Byte serializedState[], size_t size,
                                               fmi3FMUState *FMUState) {
    (void)instance;
    FmuState *st = (FmuState *)malloc(sizeof *st);
    if (!st) return fmi3Fatal;
    st->blob = (char *)malloc(size + 1);
    if (!st->blob) { free(st); return fmi3Fatal; }
    memcpy(st->blob, serializedState, size); st->blob[size] = '\0'; st->n = size;
    *FMUState = st;
    return fmi3OK;
}

/* ---- derivatives / clocks / configuration: not provided by the wrapper ---- */

FMI3_Export fmi3Status fmi3GetDirectionalDerivative(
    fmi3Instance instance, const fmi3ValueReference unknowns[], size_t nUnknowns,
    const fmi3ValueReference knowns[], size_t nKnowns, const fmi3Float64 seed[], size_t nSeed,
    fmi3Float64 sensitivity[], size_t nSensitivity) {
    (void)unknowns; (void)nUnknowns; (void)knowns; (void)nKnowns; (void)seed; (void)nSeed;
    (void)sensitivity; (void)nSensitivity;
    inst_log((Instance *)instance, fmi3Warning, "logStatusWarning",
             "maddening_fmu: directional derivatives are served by the Python sidecar API, "
             "not through the C wrapper yet");
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetAdjointDerivative(
    fmi3Instance instance, const fmi3ValueReference unknowns[], size_t nUnknowns,
    const fmi3ValueReference knowns[], size_t nKnowns, const fmi3Float64 seed[], size_t nSeed,
    fmi3Float64 sensitivity[], size_t nSensitivity) {
    (void)instance; (void)unknowns; (void)nUnknowns; (void)knowns; (void)nKnowns; (void)seed;
    (void)nSeed; (void)sensitivity; (void)nSensitivity;
    return fmi3Error;
}
/* FMI 3.0: never called on an FMU without tunable structural parameters,
 * which this one is; both used to answer fmi3OK in any state. */
static const char *const no_configuration_mode =
    "maddening_fmu: this FMU has no structural parameters, so it has no Configuration Mode "
    "(FMI 3.0: fmi3EnterConfigurationMode must not be called on such an FMU)";
FMI3_Export fmi3Status fmi3EnterConfigurationMode(fmi3Instance instance) {
    inst_log((Instance *)instance, fmi3Error, "logStatusError", no_configuration_mode);
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3ExitConfigurationMode(fmi3Instance instance) {
    inst_log((Instance *)instance, fmi3Error, "logStatusError", no_configuration_mode);
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetIntervalDecimal(fmi3Instance instance, const fmi3ValueReference vr[],
                                              size_t nvr, fmi3Float64 intervals[],
                                              fmi3IntervalQualifier qualifiers[]) {
    (void)instance; (void)vr; (void)nvr; (void)intervals; (void)qualifiers;
    return fmi3Error;   /* intervals are constant and declared in modelDescription.xml */
}
FMI3_Export fmi3Status fmi3GetIntervalFraction(fmi3Instance instance, const fmi3ValueReference vr[],
                                               size_t nvr, fmi3UInt64 counters[],
                                               fmi3UInt64 resolutions[],
                                               fmi3IntervalQualifier qualifiers[]) {
    (void)instance; (void)vr; (void)nvr; (void)counters; (void)resolutions; (void)qualifiers;
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetShiftDecimal(fmi3Instance instance, const fmi3ValueReference vr[],
                                           size_t nvr, fmi3Float64 shifts[]) {
    (void)instance; (void)vr; (void)nvr; (void)shifts; return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetShiftFraction(fmi3Instance instance, const fmi3ValueReference vr[],
                                            size_t nvr, fmi3UInt64 counters[],
                                            fmi3UInt64 resolutions[]) {
    (void)instance; (void)vr; (void)nvr; (void)counters; (void)resolutions; return fmi3Error;
}
FMI3_Export fmi3Status fmi3SetIntervalDecimal(fmi3Instance instance, const fmi3ValueReference vr[],
                                              size_t nvr, const fmi3Float64 intervals[]) {
    (void)instance; (void)vr; (void)nvr; (void)intervals; return fmi3Error;
}
FMI3_Export fmi3Status fmi3SetIntervalFraction(fmi3Instance instance, const fmi3ValueReference vr[],
                                               size_t nvr, const fmi3UInt64 counters[],
                                               const fmi3UInt64 resolutions[]) {
    (void)instance; (void)vr; (void)nvr; (void)counters; (void)resolutions; return fmi3Error;
}
FMI3_Export fmi3Status fmi3SetShiftDecimal(fmi3Instance instance, const fmi3ValueReference vr[],
                                           size_t nvr, const fmi3Float64 shifts[]) {
    (void)instance; (void)vr; (void)nvr; (void)shifts; return fmi3Error;
}
FMI3_Export fmi3Status fmi3SetShiftFraction(fmi3Instance instance, const fmi3ValueReference vr[],
                                            size_t nvr, const fmi3UInt64 counters[],
                                            const fmi3UInt64 resolutions[]) {
    (void)instance; (void)vr; (void)nvr; (void)counters; (void)resolutions; return fmi3Error;
}
/* FMI 3.0 allows fmi3EvaluateDiscreteStates only on an FMU whose description
 * declares providesEvaluateDiscreteStates, and fmi3UpdateDiscreteStates only
 * in Event Mode; this FMU has neither, and both used to answer fmi3OK in
 * any state.  fmi3UpdateDiscreteStates still fills its outputs, with the
 * values that ask for nothing. */
FMI3_Export fmi3Status fmi3EvaluateDiscreteStates(fmi3Instance instance) {
    inst_log((Instance *)instance, fmi3Error, "logStatusError",
             "maddening_fmu: fmi3EvaluateDiscreteStates is not provided (the model "
             "description does not declare providesEvaluateDiscreteStates)");
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3UpdateDiscreteStates(
    fmi3Instance instance, fmi3Boolean *discreteStatesNeedUpdate, fmi3Boolean *terminateSimulation,
    fmi3Boolean *nominalsOfContinuousStatesChanged, fmi3Boolean *valuesOfContinuousStatesChanged,
    fmi3Boolean *nextEventTimeDefined, fmi3Float64 *nextEventTime) {
    *discreteStatesNeedUpdate = fmi3False; *terminateSimulation = fmi3False;
    *nominalsOfContinuousStatesChanged = fmi3False; *valuesOfContinuousStatesChanged = fmi3False;
    *nextEventTimeDefined = fmi3False; *nextEventTime = 0.0;
    inst_log((Instance *)instance, fmi3Error, "logStatusError",
             "maddening_fmu: fmi3UpdateDiscreteStates is allowed only in Event Mode, which this "
             "FMU does not have (hasEventMode=\"false\")");
    return fmi3Error;
}

/* ---- model exchange entry points: not supported (co-simulation FMU) ---- */

FMI3_Export fmi3Status fmi3EnterContinuousTimeMode(fmi3Instance instance) { (void)instance; return fmi3Error; }
FMI3_Export fmi3Status fmi3CompletedIntegratorStep(fmi3Instance instance,
                                                   fmi3Boolean noSetFMUStatePriorToCurrentPoint,
                                                   fmi3Boolean *enterEventMode,
                                                   fmi3Boolean *terminateSimulation) {
    (void)instance; (void)noSetFMUStatePriorToCurrentPoint;
    *enterEventMode = fmi3False; *terminateSimulation = fmi3False;
    return fmi3Error;
}
/* Model exchange only: a co-simulation instance's time moves with
 * fmi3DoStep.  It used to answer fmi3OK and overwrite the wrapper's clock,
 * which fmi3DoStep reports as lastSuccessfulTime when it fails. */
FMI3_Export fmi3Status fmi3SetTime(fmi3Instance instance, fmi3Float64 time) {
    (void)time;
    inst_log((Instance *)instance, fmi3Error, "logStatusError",
             "maddening_fmu: fmi3SetTime is a model-exchange function; this is a "
             "co-simulation FMU, whose time moves with fmi3DoStep");
    return fmi3Error;
}
FMI3_Export fmi3Status fmi3SetContinuousStates(fmi3Instance instance, const fmi3Float64 x[],
                                               size_t nx) {
    (void)instance; (void)x; (void)nx; return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetContinuousStateDerivatives(fmi3Instance instance,
                                                         fmi3Float64 derivatives[], size_t nx) {
    (void)instance; (void)derivatives; (void)nx; return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetEventIndicators(fmi3Instance instance, fmi3Float64 eventIndicators[],
                                              size_t ni) {
    (void)instance; (void)eventIndicators; (void)ni; return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetContinuousStates(fmi3Instance instance, fmi3Float64 x[], size_t nx) {
    (void)instance; (void)x; (void)nx; return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetNominalsOfContinuousStates(fmi3Instance instance,
                                                         fmi3Float64 nominals[], size_t nx) {
    (void)instance; (void)nominals; (void)nx; return fmi3Error;
}
FMI3_Export fmi3Status fmi3GetNumberOfEventIndicators(fmi3Instance instance, size_t *n) {
    (void)instance; *n = 0; return fmi3OK;
}
FMI3_Export fmi3Status fmi3GetNumberOfContinuousStates(fmi3Instance instance, size_t *n) {
    (void)instance; *n = 0; return fmi3OK;
}

/* ---- co-simulation ---- */

/* FMI 3.0 allows fmi3EnterStepMode only from Event Mode, which this FMU
 * does not have: fmi3ExitInitializationMode enters Step Mode itself.  It
 * used to answer fmi3OK in any state, fmi3Terminate's Terminated too. */
FMI3_Export fmi3Status fmi3EnterStepMode(fmi3Instance instance) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    (void)phase_allows(in, 0u, "fmi3EnterStepMode",
                       "from Event Mode only, which this FMU does not have "
                       "(hasEventMode=\"false\"); fmi3ExitInitializationMode enters Step Mode");
    return fmi3Error;
}

FMI3_Export fmi3Status fmi3GetOutputDerivatives(fmi3Instance instance,
                                                const fmi3ValueReference vr[], size_t nvr,
                                                const fmi3Int32 orders[], fmi3Float64 values[],
                                                size_t nValues) {
    (void)instance; (void)vr; (void)nvr; (void)orders; (void)values; (void)nValues;
    return fmi3Error;
}

FMI3_Export fmi3Status fmi3DoStep(
    fmi3Instance instance, fmi3Float64 currentCommunicationPoint,
    fmi3Float64 communicationStepSize, fmi3Boolean noSetFMUStatePriorToCurrentPoint,
    fmi3Boolean *eventHandlingNeeded, fmi3Boolean *terminateSimulation,
    fmi3Boolean *earlyReturn, fmi3Float64 *lastSuccessfulTime) {
    (void)noSetFMUStatePriorToCurrentPoint;
    *eventHandlingNeeded = fmi3False; *terminateSimulation = fmi3False; *earlyReturn = fmi3False;
    Instance *in = (Instance *)instance;
    if (!in) { *lastSuccessfulTime = 0.0; return fmi3Error; }
    if (!phase_allows(in, IN_STEP_MODE, "fmi3DoStep",
                      "in Step Mode only (after fmi3EnterInitializationMode and "
                      "fmi3ExitInitializationMode)")) {
        *lastSuccessfulTime = in->time;
        return fmi3Error;
    }
    if (!isfinite(currentCommunicationPoint) || !isfinite(communicationStepSize)) {
        /* %.17g would write "nan" / "inf", which is not JSON */
        inst_log(in, fmi3Error, "logStatusError",
                 "maddening_fmu: communication point and step size must be finite");
        *lastSuccessfulTime = in->time;
        return fmi3Error;
    }
    if (req_reserve(in, 128)) { *lastSuccessfulTime = in->time; return fmi3Fatal; }
    c_snprintf(in, in->req, in->req_cap, "{\"op\":\"step\",\"t\":%.17g,\"dt\":%.17g}",
               currentCommunicationPoint, communicationStepSize);
    fmi3Status st = bridge_call(in, in->req);
    if (st != fmi3OK) {
        *lastSuccessfulTime = in->time;
        return st;
    }
    in->time = currentCommunicationPoint + communicationStepSize;
    *eventHandlingNeeded = fmi3False; *terminateSimulation = fmi3False;
    *earlyReturn = fmi3False; *lastSuccessfulTime = in->time;
    return fmi3OK;
}

FMI3_Export fmi3Status fmi3ActivateModelPartition(fmi3Instance instance,
                                                  fmi3ValueReference clockReference,
                                                  fmi3Float64 activationTime) {
    (void)instance; (void)clockReference; (void)activationTime;
    return fmi3Error;
}
