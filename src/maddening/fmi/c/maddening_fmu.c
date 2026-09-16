/*
 * MADDENING FMU C wrapper (FMI 3.0 co-simulation).
 *
 * The FMU binary owns no simulation state: every FMI call is marshalled
 * into a small JSON message and forwarded over a TCP socket to a Python
 * sidecar (maddening.fmi.tcp_bridge.FmuTcpBridge) that holds the
 * JAX-JITted graph.  Wire format: 4-byte big-endian length prefix + one
 * UTF-8 JSON object, both directions.
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

/* Send `req` (JSON text) and store the JSON reply in in->resp.  Returns
 * fmi3OK when the reply carries "ok":true, fmi3Error otherwise (with the
 * server's error text logged). */
static fmi3Status bridge_call(Instance *in, const char *req) {
    unsigned char head[4];
    size_t n = strlen(req);
    head[0] = (unsigned char)(n >> 24); head[1] = (unsigned char)(n >> 16);
    head[2] = (unsigned char)(n >> 8);  head[3] = (unsigned char)n;
    if (in->resp) in->resp[0] = '\0';
    if (send_all(in->sock, (const char *)head, 4) || send_all(in->sock, req, n)) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: send failed");
        return fmi3Error;
    }
    if (recv_all(in->sock, (char *)head, 4)) {
        inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: recv failed");
        return fmi3Error;
    }
    size_t m = ((size_t)head[0] << 24) | ((size_t)head[1] << 16)
             | ((size_t)head[2] << 8) | (size_t)head[3];
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
    if (strstr(in->resp, "\"ok\":true") == NULL) {
        const char *e = strstr(in->resp, "\"error\":\"");
        inst_log(in, fmi3Error, "logStatusError", e ? e + 9 : in->resp);
        return fmi3Error;
    }
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

/* Build {"op":"set","vr":[..],"values":[..]} from any numeric width. */
static fmi3Status do_set(Instance *in, const fmi3ValueReference vr[], size_t nvr,
                         const double values[], size_t nvalues) {
    if (req_reserve(in, 64 + 24 * nvr + 32 * nvalues)) return fmi3Fatal;
    char *w = in->req;
    w += sprintf(w, "{\"op\":\"set\",\"vr\":[");
    for (size_t i = 0; i < nvr; ++i) w += sprintf(w, "%s%u", i ? "," : "", (unsigned)vr[i]);
    w += sprintf(w, "],\"values\":[");
    for (size_t i = 0; i < nvalues; ++i) {
        double v = values[i];
        if (isnan(v) || isinf(v)) {
            inst_log(in, fmi3Error, "logStatusError", "maddening_fmu: non-finite value");
            return fmi3Error;
        }
        w += sprintf(w, "%s%.17g", i ? "," : "", v);
    }
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
    return parse_values(in, out, nvalues);
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
    if (bridge_call(in, "{\"op\":\"hello\"}") != fmi3OK) {
        sock_close(in->sock); free(in->req); free(in->resp); free(in); return NULL;
    }
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

/* ---- FMU state: an opaque base64 blob produced by the sidecar ---- */

typedef struct { char *blob; size_t n; } FmuState;

FMI3_Export fmi3Status fmi3GetFMUState(fmi3Instance instance, fmi3FMUState *FMUState) {
    Instance *in = (Instance *)instance;
    if (!in) return fmi3Error;
    if (bridge_call(in, "{\"op\":\"get_state\"}") != fmi3OK) return fmi3Error;
    if (in->resp == NULL) return fmi3Error;
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
