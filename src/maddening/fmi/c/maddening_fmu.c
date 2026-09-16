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
 * The endpoint is read from "<resourcePath>/endpoint.txt" ("host:port"),
 * or from the MADDENING_FMU_ENDPOINT environment variable.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
#  include <sys/socket.h>
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
#define FRAME_MAX_BINARY (64ul * 1024ul * 1024ul)   /* the bridge's frame limit */
#define HDR_MAX 4096                         /* a binary reply's JSON header */

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
} Instance;

/* ------------------------------------------------------------------ util */

static void inst_log(Instance *in, fmi3Status status, const char *category,
                     const char *msg) {
    if (in && in->log) {
        in->log(in->env, status, category, msg);
    }
}

/* A sidecar that went away must surface as fmi3Error from the next call,
 * never as SIGPIPE killing the importer's process. */
#ifdef MSG_NOSIGNAL
#  define SEND_FLAGS MSG_NOSIGNAL
#else
#  define SEND_FLAGS 0
#endif

static int send_all(sock_t s, const char *buf, size_t n) {
    while (n > 0) {
        ssize_t k = send(s, buf, n, SEND_FLAGS);
        if (k <= 0) return -1;
        buf += k; n -= (size_t)k;
    }
    return 0;
}

static int recv_all(sock_t s, char *buf, size_t n) {
    while (n > 0) {
        ssize_t k = recv(s, buf, n, 0);
        if (k <= 0) return -1;
        buf += k; n -= (size_t)k;
    }
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

/* Send one frame (`n` bytes of `req`; `binary` sets bit 31 of the
 * prefix) and store the reply in in->resp.  A JSON reply must carry
 * "ok":true.  A binary reply is split: its JSON header is copied,
 * NUL-terminated, into in->hdr (which must carry "ok":true) and
 * in->raw / in->raw_len point at the raw part inside in->resp.  Returns
 * fmi3OK or fmi3Error (with the server's error text logged). */
static fmi3Status bridge_xfer(Instance *in, const char *req, size_t n, int binary) {
    unsigned char head[4];
    in->resp_binary = 0; in->raw = NULL; in->raw_len = 0; in->hdr[0] = '\0';
    if (in->resp) in->resp[0] = '\0';
    if (n > FRAME_LEN_MASK) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: request exceeds the frame size");
        return fmi3Error;
    }
    put_be32(head, (unsigned long)n | (binary ? FRAME_BINARY : 0ul));
    if (send_all(in->sock, (const char *)head, 4) || send_all(in->sock, req, n)) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: send failed");
        return fmi3Error;
    }
    if (recv_all(in->sock, (char *)head, 4)) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: recv failed");
        return fmi3Error;
    }
    unsigned long word = get_be32(head);
    int reply_binary = (word & FRAME_BINARY) != 0;
    size_t m = (size_t)(word & FRAME_LEN_MASK);
    if (reply_binary && m > FRAME_MAX_BINARY) {
        /* the peer is out of step with us; never allocate for it */
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: binary reply exceeds the frame limit");
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
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: recv body failed");
        return fmi3Error;
    }
    in->resp[m] = '\0';
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
 * plain non-negative decimal. */
static int hdr_count(const Instance *in, size_t *count) {
    const char *p = strstr(in->hdr, "\"n\":");
    if (!p) return -1;
    p += 4;
    while (*p == ' ') ++p;
    if (*p < '0' || *p > '9') return -1;
    char *end;
    errno = 0;
    unsigned long long v = strtoull(p, &end, 10);
    if (end == p || errno == ERANGE || v > (unsigned long long)FRAME_MAX_BINARY) return -1;
    *count = (size_t)v;
    return 0;
}

/* The raw doubles of a binary get reply: the header's count must match
 * the raw length exactly and cover the caller's array. */
static fmi3Status parse_binary_values(Instance *in, double *out, size_t n) {
    size_t have;
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
    if (n > 0) f64_from_le(out, (const unsigned char *)in->raw, n);
    return fmi3OK;
}

/* Parse "values":[n0,n1,...] from in->resp into out[0..n). */
static fmi3Status parse_values(Instance *in, double *out, size_t n) {
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
        out[i] = strtod(p, &end);
        if (end == p) {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: malformed number in reply");
            return fmi3Error;
        }
        p = end;
    }
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

/* Build a set request from any numeric width: a binary frame
 * ({"op":"set","vr":[..],"n":N,"dtype":"f64"} + N raw little-endian
 * doubles) after a protocol-2 hello, else the JSON
 * {"op":"set","vr":[..],"values":[..]} with %.17g text. */
static fmi3Status do_set(Instance *in, const fmi3ValueReference vr[], size_t nvr,
                         const double values[], size_t nvalues) {
    if (in->binary && nvalues > FRAME_MAX_BINARY / sizeof(double)) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: set exceeds the frame limit");
        return fmi3Error;
    }
    for (size_t i = 0; i < nvalues; ++i) {
        if (isnan(values[i]) || isinf(values[i])) {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: non-finite value");
            return fmi3Error;
        }
    }
    if (in->binary) {
        if (req_reserve(in, 4 + 96 + 24 * nvr + sizeof(double) * nvalues)) return fmi3Fatal;
        char *h = in->req + 4;
        char *w = h;
        w += sprintf(w, "{\"op\":\"set\",\"vr\":[");
        for (size_t i = 0; i < nvr; ++i) w += sprintf(w, "%s%u", i ? "," : "", (unsigned)vr[i]);
        w += sprintf(w, "],\"n\":%lu,\"dtype\":\"f64\"}", (unsigned long)nvalues);
        size_t hl = (size_t)(w - h);
        put_be32((unsigned char *)in->req, (unsigned long)hl);
        f64_to_le((unsigned char *)w, values, nvalues);
        return bridge_xfer(in, in->req, 4 + hl + sizeof(double) * nvalues, 1);
    }
    if (req_reserve(in, 64 + 24 * nvr + 32 * nvalues)) return fmi3Fatal;
    char *w = in->req;
    w += sprintf(w, "{\"op\":\"set\",\"vr\":[");
    for (size_t i = 0; i < nvr; ++i) w += sprintf(w, "%s%u", i ? "," : "", (unsigned)vr[i]);
    w += sprintf(w, "],\"values\":[");
    for (size_t i = 0; i < nvalues; ++i) w += sprintf(w, "%s%.17g", i ? "," : "", values[i]);
    sprintf(w, "]}");
    return bridge_call(in, in->req);
}

static fmi3Status do_get(Instance *in, const fmi3ValueReference vr[], size_t nvr,
                         double *out, size_t nvalues) {
    if (req_reserve(in, 64 + 24 * nvr)) return fmi3Fatal;
    char *w = in->req;
    w += sprintf(w, "{\"op\":\"get\",\"vr\":[");
    for (size_t i = 0; i < nvr; ++i) w += sprintf(w, "%s%u", i ? "," : "", (unsigned)vr[i]);
    sprintf(w, "]}");
    fmi3Status st = bridge_call(in, in->req);
    if (st != fmi3OK) return st;
    if (in->resp_binary) return parse_binary_values(in, out, nvalues);
    return parse_values(in, out, nvalues);
}

/* A hello reply negotiates binary frames when it is a JSON frame that
 * carries "protocol" >= 2 and "binary":true; anything else (an older
 * bridge, in particular) leaves the instance on JSON. */
static int hello_negotiated_binary(const Instance *in) {
    if (in->resp == NULL || in->resp_binary) return 0;
    const char *p = strstr(in->resp, "\"protocol\":");
    if (!p) return 0;
    long v = strtol(p + 11, NULL, 10);
    return v >= 2 && strstr(in->resp, "\"binary\":true") != NULL;
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

static sock_t connect_endpoint(const char *host, int port) {
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
        if (connect(s, ai->ai_addr, (int)ai->ai_addrlen) == 0) break;
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

    char host[256]; int port = 0;
    if (read_endpoint(resourcePath, host, sizeof host, &port)) {
        inst_log(in, fmi3Error, "logStatusError",
                 "maddening_fmu: no endpoint (resources/endpoint.txt or MADDENING_FMU_ENDPOINT)");
        free(in); return NULL;
    }
    in->sock = connect_endpoint(host, port);
    if (in->sock == SOCK_INVALID) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: cannot connect to sidecar");
        free(in); return NULL;
    }
    if (bridge_call(in, "{\"op\":\"hello\",\"protocol\":2,\"binary\":true}") != fmi3OK) {
        sock_close(in->sock); free(in->req); free(in->resp); free(in); return NULL;
    }
    in->binary = hello_negotiated_binary(in);
    if (instantiationToken && *instantiationToken) {
        char needle[300];
        snprintf(needle, sizeof needle, "\"token\":\"%s\"", instantiationToken);
        if (in->resp == NULL || strstr(in->resp, needle) == NULL) {
            inst_log(in, fmi3Error, "logStatusError",
                     "maddening_fmu: instantiation token does not match the sidecar's graph");
            sock_close(in->sock); free(in->req); free(in->resp); free(in); return NULL;
        }
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
    if (in->sock != SOCK_INVALID) {
        bridge_call(in, "{\"op\":\"terminate\"}");
        sock_close(in->sock);
    }
    free(in->req); free(in->resp); free(in);
}

FMI3_Export fmi3Status fmi3EnterInitializationMode(
    fmi3Instance instance, fmi3Boolean toleranceDefined, fmi3Float64 tolerance,
    fmi3Float64 startTime, fmi3Boolean stopTimeDefined, fmi3Float64 stopTime) {
    (void)toleranceDefined; (void)tolerance; (void)stopTimeDefined; (void)stopTime;
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    in->time = startTime;
    return fmi3OK;
}

FMI3_Export fmi3Status fmi3ExitInitializationMode(fmi3Instance instance) {
    return instance ? fmi3OK : fmi3Error;
}

FMI3_Export fmi3Status fmi3EnterEventMode(fmi3Instance instance) {
    /* modelDescription advertises hasEventMode="false" */
    (void)instance; return fmi3Error;
}

FMI3_Export fmi3Status fmi3Terminate(fmi3Instance instance) {
    Instance *in = (Instance *)instance;
    return in ? bridge_call(in, "{\"op\":\"terminate\"}") : fmi3Error;
}

FMI3_Export fmi3Status fmi3Reset(fmi3Instance instance) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    in->time = 0.0;
    return bridge_call(in, "{\"op\":\"reset\"}");
}

/* ---- getters: every numeric width goes through double ---- */

#define DEFINE_GET(NAME, CTYPE)                                                        \
FMI3_Export fmi3Status NAME(fmi3Instance instance, const fmi3ValueReference vr[],       \
                            size_t nvr, CTYPE values[], size_t nValues) {              \
    Instance *in = (Instance *)instance;                                               \
    if (!in) return fmi3Error;                                                         \
    if (nValues == 0) return fmi3OK;                                                   \
    double *tmp = (double *)malloc(nValues * sizeof(double));                          \
    if (!tmp) return fmi3Fatal;                                                        \
    fmi3Status st = do_get(in, vr, nvr, tmp, nValues);                                 \
    if (st == fmi3OK) for (size_t i = 0; i < nValues; ++i) values[i] = (CTYPE)tmp[i];  \
    free(tmp);                                                                         \
    return st;                                                                         \
}

#define DEFINE_SET(NAME, CTYPE)                                                        \
FMI3_Export fmi3Status NAME(fmi3Instance instance, const fmi3ValueReference vr[],       \
                            size_t nvr, const CTYPE values[], size_t nValues) {        \
    Instance *in = (Instance *)instance;                                               \
    if (!in) return fmi3Error;                                                         \
    if (nValues == 0) return fmi3OK;                                                   \
    double *tmp = (double *)malloc(nValues * sizeof(double));                          \
    if (!tmp) return fmi3Fatal;                                                        \
    for (size_t i = 0; i < nValues; ++i) tmp[i] = (double)values[i];                   \
    fmi3Status st = do_set(in, vr, nvr, tmp, nValues);                                 \
    free(tmp);                                                                         \
    return st;                                                                         \
}

DEFINE_GET(fmi3GetFloat32, fmi3Float32)
DEFINE_GET(fmi3GetFloat64, fmi3Float64)
DEFINE_GET(fmi3GetInt8,    fmi3Int8)
DEFINE_GET(fmi3GetUInt8,   fmi3UInt8)
DEFINE_GET(fmi3GetInt16,   fmi3Int16)
DEFINE_GET(fmi3GetUInt16,  fmi3UInt16)
DEFINE_GET(fmi3GetInt32,   fmi3Int32)
DEFINE_GET(fmi3GetUInt32,  fmi3UInt32)
DEFINE_GET(fmi3GetInt64,   fmi3Int64)
DEFINE_GET(fmi3GetUInt64,  fmi3UInt64)
DEFINE_GET(fmi3GetBoolean, fmi3Boolean)

DEFINE_SET(fmi3SetFloat32, fmi3Float32)
DEFINE_SET(fmi3SetFloat64, fmi3Float64)
DEFINE_SET(fmi3SetInt8,    fmi3Int8)
DEFINE_SET(fmi3SetUInt8,   fmi3UInt8)
DEFINE_SET(fmi3SetInt16,   fmi3Int16)
DEFINE_SET(fmi3SetUInt16,  fmi3UInt16)
DEFINE_SET(fmi3SetInt32,   fmi3Int32)
DEFINE_SET(fmi3SetUInt32,  fmi3UInt32)
DEFINE_SET(fmi3SetInt64,   fmi3Int64)
DEFINE_SET(fmi3SetUInt64,  fmi3UInt64)
DEFINE_SET(fmi3SetBoolean, fmi3Boolean)

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
FMI3_Export fmi3Status fmi3GetClock(fmi3Instance instance, const fmi3ValueReference vr[],
                                    size_t nvr, fmi3Clock values[]) {
    (void)instance;
    for (size_t i = 0; i < nvr; ++i) { (void)vr[i]; values[i] = fmi3ClockInactive; }
    return fmi3OK;
}
FMI3_Export fmi3Status fmi3SetClock(fmi3Instance instance, const fmi3ValueReference vr[],
                                    size_t nvr, const fmi3Clock values[]) {
    (void)instance; (void)vr; (void)nvr; (void)values;
    return fmi3OK;   /* constant-interval clocks: ticks are implied by time */
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

FMI3_Export fmi3Status fmi3GetFMUState(fmi3Instance instance, fmi3FMUState *FMUState) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    if (bridge_call(in, "{\"op\":\"get_state\"}") != fmi3OK) return fmi3Error;
    if (in->resp == NULL) return fmi3Error;
    if (in->resp_binary) {
        size_t n;
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
    if (in->binary) {
        /* length-delimited raw bytes: nothing the importer supplies can
         * break the framing, the bridge validates the archive itself */
        if (st->n > FRAME_MAX_BINARY) {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: FMU state exceeds the frame limit");
            return fmi3Error;
        }
        char hdr[64];
        int hl = snprintf(hdr, sizeof hdr, "{\"op\":\"set_state\",\"n\":%lu}", (unsigned long)st->n);
        if (hl <= 0 || (size_t)hl >= sizeof hdr) return fmi3Error;
        if (req_reserve(in, 4 + (size_t)hl + st->n)) return fmi3Fatal;
        put_be32((unsigned char *)in->req, (unsigned long)hl);
        memcpy(in->req + 4, hdr, (size_t)hl);
        memcpy(in->req + 4 + hl, st->blob, st->n);
        return bridge_xfer(in, in->req, 4 + (size_t)hl + st->n, 1);
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
    return bridge_call(in, in->req);
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
FMI3_Export fmi3Status fmi3EnterConfigurationMode(fmi3Instance instance) { (void)instance; return fmi3OK; }
FMI3_Export fmi3Status fmi3ExitConfigurationMode(fmi3Instance instance) { (void)instance; return fmi3OK; }
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
FMI3_Export fmi3Status fmi3EvaluateDiscreteStates(fmi3Instance instance) { (void)instance; return fmi3OK; }
FMI3_Export fmi3Status fmi3UpdateDiscreteStates(
    fmi3Instance instance, fmi3Boolean *discreteStatesNeedUpdate, fmi3Boolean *terminateSimulation,
    fmi3Boolean *nominalsOfContinuousStatesChanged, fmi3Boolean *valuesOfContinuousStatesChanged,
    fmi3Boolean *nextEventTimeDefined, fmi3Float64 *nextEventTime) {
    (void)instance;
    *discreteStatesNeedUpdate = fmi3False; *terminateSimulation = fmi3False;
    *nominalsOfContinuousStatesChanged = fmi3False; *valuesOfContinuousStatesChanged = fmi3False;
    *nextEventTimeDefined = fmi3False; *nextEventTime = 0.0;
    return fmi3OK;
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
FMI3_Export fmi3Status fmi3SetTime(fmi3Instance instance, fmi3Float64 time) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    in->time = time;
    return fmi3OK;
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

FMI3_Export fmi3Status fmi3EnterStepMode(fmi3Instance instance) { (void)instance; return fmi3OK; }

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
    if (req_reserve(in, 128)) { *lastSuccessfulTime = in->time; return fmi3Fatal; }
    sprintf(in->req, "{\"op\":\"step\",\"t\":%.17g,\"dt\":%.17g}",
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
