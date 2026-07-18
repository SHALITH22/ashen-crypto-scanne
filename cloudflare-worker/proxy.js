/*
 * BINANCE_PROXY_URL relay - referenced by scanner/data.py's _proxy_get()
 * but never actually deployed anywhere, which is why scan_health.json has
 * shown proxy_stats.configured: false on every run: main.py checks for the
 * env var, finds it unset, and every long-tail (~371 pairs not on
 * Binance.US) call falls straight through to no_data. long_tail.enabled
 * is also off in config/settings.yaml for the same reason - no point
 * attempting 371 pairs' worth of guaranteed-blocked calls every scan.
 *
 * What this does: relays GET requests to Binance's real global futures
 * API (fapi.binance.com) using Cloudflare's own edge IPs, which are not
 * geo-blocked the way GitHub Actions' IPs are (see scanner/data.py's
 * module docstring for the full blocking story). scanner/data.py only
 * ever calls two paths - /fapi/v1/klines and /fapi/v1/ticker/price - both
 * passed straight through here unmodified.
 *
 * Deploy (Cloudflare's free tier - 100k requests/day, no billing needed):
 *   1. workers.cloudflare.com -> sign in / create a free account.
 *   2. Create application -> Create Worker -> give it any name (e.g.
 *      "binance-relay") -> Deploy (starts from a placeholder - that's fine).
 *   3. Edit code -> replace the placeholder with this entire file -> Deploy.
 *   4. Settings -> Variables and Secrets -> add a secret named
 *      PROXY_SECRET with any long random value you choose - this is what
 *      stops random internet traffic from using your worker as an open
 *      Binance relay.
 *   5. Copy the worker's URL (shown on its overview page, looks like
 *      https://binance-relay.<your-subdomain>.workers.dev).
 *   6. In the ashen-crypto-scanne GitHub repo: Settings -> Secrets and
 *      variables -> Actions -> New repository secret, twice:
 *        BINANCE_PROXY_URL = the worker URL from step 5 (no trailing slash)
 *        BINANCE_PROXY_SECRET = the same value you set as PROXY_SECRET
 *   7. In config/settings.yaml, flip long_tail.enabled to true - the next
 *      scan run will pick up all ~371 additional pairs automatically, no
 *      other code changes needed.
 *
 * Cost/limits: Cloudflare Workers' free tier is 100k requests/day - the
 * long_tail feature's own docstring in main.py already sized itself to
 * ~80k/day at a 20-minute scan cadence with 2 timeframes x ~371 pairs, so
 * this fits comfortably as long as the scan cadence doesn't get much
 * faster than that without re-checking the math.
 */

const UPSTREAM = "https://fapi.binance.com";
const ALLOWED_PATHS = new Set(["/fapi/v1/klines", "/fapi/v1/ticker/price"]);

export default {
  async fetch(request, env) {
    if (request.method !== "GET") {
      return new Response("Method not allowed", { status: 405 });
    }

    const url = new URL(request.url);

    if (!ALLOWED_PATHS.has(url.pathname)) {
      return new Response("Not found", { status: 404 });
    }

    if (env.PROXY_SECRET) {
      const provided = request.headers.get("X-Proxy-Secret");
      if (provided !== env.PROXY_SECRET) {
        return new Response("Unauthorized", { status: 401 });
      }
    }

    const upstreamUrl = `${UPSTREAM}${url.pathname}${url.search}`;

    try {
      const upstreamResp = await fetch(upstreamUrl, {
        method: "GET",
        headers: { "Accept": "application/json" },
      });
      const body = await upstreamResp.text();
      return new Response(body, {
        status: upstreamResp.status,
        headers: { "Content-Type": "application/json" },
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: String(err) }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }
  },
};
