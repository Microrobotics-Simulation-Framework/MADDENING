"""Key derivation: domain separation, and what a weak token costs.

The CURVE keys are blake2b(role + b"\\x00" + token, person=b"maddening-curve").
Both role strings are fixed literals with no NUL in them and they differ
at byte 0, so no (role, token) pair can collide with another -- checked
below by construction and by search.

The consequence that is NOT stated anywhere: the derivation is a single
unsalted, uniterated hash of a human-chosen secret, and
``TransportAuth`` (unlike ``APIAuth``) cannot generate a token -- both
ends must hold the same one, so it is always operator-chosen.  Nothing
enforces a minimum length or entropy: a one-character token is accepted.
Anyone who learns the CURVE *server public key* can therefore recover
the token offline at the rate measured below, and that token is also the
HTTP bearer credential.
"""
import hashlib, itertools, string, time
from maddening.api.auth import APIAuth
from maddening.transport_auth import TransportAuth, TransportAuthError


def domain_separation():
    seen = {}
    collisions = []
    for tok in ["", "a", "server", "client", "\x00", "server\x00x", "client\x00x",
                "x" * 100, "the-shared-token"]:
        if not tok.strip():
            continue
        auth = TransportAuth(token=tok)
        for role, kp in (("server", auth.server_keypair()),
                         ("client", auth.client_keypair())):
            if kp in seen:
                collisions.append((seen[kp], (role, tok)))
            seen[kp] = (role, tok)
    print(f"  distinct keypairs from {len(seen)} (role, token) pairs, "
          f"collisions: {collisions}")


def weak_tokens_accepted():
    for tok in ["a", "1", "password", "x" * 1000]:
        TransportAuth(token=tok).server_keypair()
        APIAuth(bind_host="0.0.0.0", token=tok, environ={})
        print(f"  token {tok[:20]!r:<24} accepted by BOTH TransportAuth and APIAuth")
    for tok in ["", "   "]:
        try:
            TransportAuth(token=tok)
            print(f"  token {tok!r} ACCEPTED (unexpected)")
        except TransportAuthError:
            print(f"  token {tok!r:<24} refused (blank is a configuration error)")


def offline_guess_rate():
    """Candidate tokens per second, given the server's CURVE public key."""
    import zmq
    from zmq.utils import z85
    target, _ = TransportAuth(token="hunter2").server_keypair()

    def derive(tok: str) -> bytes:
        scalar = hashlib.blake2b(b"server\x00" + tok.encode(),
                                 digest_size=32, person=b"maddening-curve").digest()
        return zmq.curve_public(z85.encode(scalar))

    words = ["".join(c) for c in itertools.product(string.ascii_lowercase, repeat=3)]
    t0 = time.perf_counter()
    hits = [w for w in words if derive(w) == target]
    rate = len(words) / (time.perf_counter() - t0)
    print(f"  offline candidate rate: {rate:,.0f} tokens/s single-threaded "
          f"(pure Python + libsodium scalarmult); hits in a 17,576-word list: {hits}")
    t0 = time.perf_counter()
    found = derive("hunter2") == target
    print(f"  a known candidate is confirmed in {(time.perf_counter()-t0)*1e6:.0f} us: {found}")


if __name__ == "__main__":
    print("Domain separation:")
    domain_separation()
    print("\nWeak / blank tokens:")
    weak_tokens_accepted()
    print("\nOffline search, given the server public key:")
    offline_guess_rate()
