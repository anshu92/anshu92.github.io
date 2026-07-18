# Synaptic Radio Article Likes Worker

Cloudflare Worker + KV backend for global article like counts.

## Deploy

1. Install Wrangler if needed:

```sh
npm install --save-dev wrangler
```

2. Create the KV namespace:

```sh
npx wrangler kv namespace create LIKES
```

3. Copy the generated namespace `id` into `wrangler.toml`.

4. Deploy:

```sh
npx wrangler deploy
```

5. Set the deployed Worker URL in the site config:

```toml
[params.likes]
workerURL = "https://synaptic-radio-article-likes.<your-subdomain>.workers.dev"
```

The Worker exposes:

- `GET /likes?ids=/post/example/,/post/other/`
- `POST /likes` with `{"id":"/post/example/","action":"like"}`
- `POST /likes` with `{"id":"/post/example/","action":"unlike"}`

Counts are global. Each browser still stores its own liked/unliked state in localStorage so a reader can toggle their like without needing an account.
