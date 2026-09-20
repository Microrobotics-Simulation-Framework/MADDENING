/*
 * Bearer-token plumbing for the bundled MADDENING UIs.
 *
 * Served at /viz/auth.js, without a credential, because it contains no
 * credential: it FINDS a token, it never carries one.  A loopback-bound
 * server needs no token at all and this file then does nothing visible.
 *
 * Where the token comes from, in order:
 *   1. #token=... on this page's URL.  A fragment is never sent to the
 *      server, so this delivery channel reaches no access log at all.
 *      It is the form the server prints at start-up.
 *   2. ?token=... on this page's URL, accepted because a query string
 *      survives being pasted through tools that drop fragments -- but a
 *      query string DOES reach the server's access log, so prefer (1).
 *      (No authenticated route accepts ?token=; this is the page URL,
 *      which is served without a credential in the first place.)
 *      Either form is removed from the address bar immediately with
 *      history.replaceState, so it does not sit in the browser history,
 *      get copied out of the URL bar, or leak through a Referer header
 *      to anything the page later loads.
 *   3. sessionStorage, so a reload keeps working.  Per-tab and cleared
 *      when the tab closes; localStorage would outlive the session.
 *   4. a prompt, shown the first time a request comes back 401.
 *
 * How it is presented:
 *   - fetch(): window.fetch is wrapped so every same-origin request
 *     carries "Authorization: Bearer <token>".  Wrapping rather than
 *     renaming keeps the three pages' existing call sites unchanged --
 *     a page that adds a call later is covered automatically instead of
 *     failing whenever somebody forgets the helper.
 *   - WebSocket: browsers cannot set a header on the handshake, so the
 *     token rides in a subprotocol.  mdWebSocket() builds it.  The
 *     token is NOT put in the WebSocket URL: a query string reaches the
 *     server's access log, and this does not.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "maddening.api.token";
  var WS_SUBPROTOCOL = "maddening.v1";
  var WS_BEARER_PREFIX = "maddening.bearer.";

  var token = "";
  var prompting = null;

  function readStored() {
    try {
      return window.sessionStorage.getItem(STORAGE_KEY) || "";
    } catch (err) {
      return "";   // private mode, or storage disabled
    }
  }

  function store(value) {
    try {
      window.sessionStorage.setItem(STORAGE_KEY, value);
    } catch (err) {
      /* keep it in memory only */
    }
  }

  /* 1 and 2. #token=... or ?token=... , then scrub both out of the URL. */
  try {
    var hash = new URLSearchParams((window.location.hash || "").replace(/^#/, ""));
    var query = new URLSearchParams(window.location.search || "");
    var fromUrl = hash.get("token") || query.get("token");
    if (fromUrl) {
      token = fromUrl;
      store(token);
      hash.delete("token");
      query.delete("token");
      var q = query.toString();
      var h = hash.toString();
      window.history.replaceState(
        null, "",
        window.location.pathname + (q ? "?" + q : "") + (h ? "#" + h : "")
      );
    }
  } catch (err) {
    /* no URLSearchParams / no history: fall through to storage */
  }

  /* 3. whatever a previous load put in sessionStorage. */
  if (!token) {
    token = readStored();
  }

  /* base64url(token), unpadded -- matches maddening.api.auth.encode_ws_bearer.
     RFC 6455 subprotocol names are HTTP tokens, so an operator-chosen
     MADDENING_API_TOKEN containing a comma or a space cannot travel raw. */
  function base64url(value) {
    var bytes = new TextEncoder().encode(value);
    var binary = "";
    for (var i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary)
      .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function sameOrigin(url) {
    try {
      return new URL(url, window.location.href).origin === window.location.origin;
    } catch (err) {
      return false;
    }
  }

  /* 4. ask, once, and only once even if twenty requests 401 at once. */
  function askForToken() {
    if (prompting) {
      return prompting;
    }
    prompting = new Promise(function (resolve) {
      var answer = window.prompt(
        "This MADDENING server requires an API token.\n\n" +
        "It was printed in the server log at start-up, or is the value " +
        "of MADDENING_API_TOKEN.\n\n" +
        "You can also open this page as  " + window.location.pathname +
        "#token=<token>  to skip this prompt. A fragment is never sent " +
        "to the server, so it reaches no access log.",
        ""
      );
      if (answer && answer.trim()) {
        token = answer.trim();
        store(token);
        resolve(true);
      } else {
        resolve(false);
      }
    });
    prompting.then(function () { prompting = null; });
    return prompting;
  }

  function withBearer(init) {
    var next = {};
    for (var key in init) {
      if (Object.prototype.hasOwnProperty.call(init, key)) {
        next[key] = init[key];
      }
    }
    var headers = new Headers(init.headers || undefined);
    headers.set("Authorization", "Bearer " + token);
    next.headers = headers;
    return next;
  }

  var nativeFetch = window.fetch.bind(window);

  window.fetch = function (input, init) {
    init = init || {};
    var url = (typeof input === "string") ? input : (input && input.url) || "";
    if (!sameOrigin(url)) {
      return nativeFetch(input, init);
    }
    var attempt = token ? withBearer(init) : init;
    return nativeFetch(input, attempt).then(function (response) {
      if (response.status !== 401) {
        return response;
      }
      // Retried exactly once, with a token the user just supplied: a
      // loop here would prompt forever against a wrong token.
      return askForToken().then(function (got) {
        if (!got) {
          return response;
        }
        return nativeFetch(input, withBearer(init));
      });
    });
  };

  /* WebSocket. Offers two subprotocols; the server selects the second,
     as RFC 6455 requires it to select one of the offered names. */
  function protocols() {
    return token ? [WS_BEARER_PREFIX + base64url(token), WS_SUBPROTOCOL]
                 : [WS_SUBPROTOCOL];
  }

  window.mdWebSocket = function (url) {
    return new WebSocket(url, protocols());
  };

  window.maddeningAuth = {
    token: function () { return token; },
    protocols: protocols,
    prompt: askForToken,
    /* Append #token= to a URL a user is meant to copy elsewhere.  The
       fragment form, so the credential never reaches the server's log. */
    shareableUrl: function (path) {
      var base = window.location.origin + (path || window.location.pathname);
      return token ? base + "#token=" + encodeURIComponent(token) : base;
    }
  };
})();
