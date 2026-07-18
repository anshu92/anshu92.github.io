const DEFAULT_ALLOWED_ORIGINS = [
  "https://anshu92.github.io",
  "http://localhost:1313",
  "http://127.0.0.1:1313",
];

const MAX_IDS_PER_REQUEST = 100;
const MAX_ID_LENGTH = 240;

function allowedOrigins(env) {
  const configured = (env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);

  return configured.length > 0 ? configured : DEFAULT_ALLOWED_ORIGINS;
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const allowed = allowedOrigins(env);
  const allowOrigin = allowed.includes(origin) ? origin : allowed[0];

  return {
    "Access-Control-Allow-Origin": allowOrigin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
  };
}

function json(request, env, body, init = {}) {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...corsHeaders(request, env),
      ...(init.headers || {}),
    },
  });
}

function normalizeArticleId(id) {
  if (typeof id !== "string") return "";

  const trimmed = id.trim();
  if (!trimmed || trimmed.length > MAX_ID_LENGTH) return "";
  if (!trimmed.startsWith("/post/")) return "";

  return trimmed;
}

function keyForArticle(id) {
  return `article-like-count:${id}`;
}

async function readCount(env, id) {
  const value = await env.LIKES.get(keyForArticle(id));
  const count = Number.parseInt(value || "0", 10);
  return Number.isFinite(count) && count > 0 ? count : 0;
}

async function writeCount(env, id, count) {
  await env.LIKES.put(keyForArticle(id), String(Math.max(0, count)));
}

async function handleGet(request, env) {
  const url = new URL(request.url);
  const ids = url.searchParams
    .getAll("id")
    .concat((url.searchParams.get("ids") || "").split(","))
    .map(normalizeArticleId)
    .filter(Boolean)
    .slice(0, MAX_IDS_PER_REQUEST);

  const uniqueIds = [...new Set(ids)];
  const counts = {};

  await Promise.all(
    uniqueIds.map(async (id) => {
      counts[id] = await readCount(env, id);
    }),
  );

  return json(request, env, { counts });
}

async function handlePost(request, env) {
  let payload;
  try {
    payload = await request.json();
  } catch {
    return json(request, env, { error: "Invalid JSON body." }, { status: 400 });
  }

  const id = normalizeArticleId(payload.id);
  if (!id) {
    return json(request, env, { error: "Invalid article id." }, { status: 400 });
  }

  const action = payload.action === "unlike" ? "unlike" : "like";
  const current = await readCount(env, id);
  const next = action === "like" ? current + 1 : Math.max(0, current - 1);

  await writeCount(env, id, next);

  return json(request, env, { id, count: next, liked: action === "like" });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request, env) });
    }

    const url = new URL(request.url);
    if (url.pathname !== "/" && url.pathname !== "/likes") {
      return json(request, env, { error: "Not found." }, { status: 404 });
    }

    if (request.method === "GET") return handleGet(request, env);
    if (request.method === "POST") return handlePost(request, env);

    return json(request, env, { error: "Method not allowed." }, { status: 405 });
  },
};
