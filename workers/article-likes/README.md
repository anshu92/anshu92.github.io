# Synaptic Radio Article Likes Worker

Cloudflare Worker + KV backend for global article like counts.

## Deploy

1. Install Wrangler:

```sh
npm install
```

2. Create the KV namespace:

```sh
npm run kv:create
```

3. Copy the generated namespace `id` into `wrangler.toml`.

4. Deploy:

```sh
npm run deploy
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

## Network Notes

If `npm install` or `npx wrangler ...` times out while reaching `https://registry.npmjs.org/`, the issue is local network/proxy access to npm, not the Worker code. Configure npm for the network you are on, then retry:

```sh
npm config get registry
npm config get proxy
npm config get https-proxy
npm ping
```

If your network requires a proxy, set it explicitly:

```sh
npm config set proxy http://proxy-host:proxy-port
npm config set https-proxy http://proxy-host:proxy-port
```

If your company provides an internal npm registry, use that registry instead:

```sh
npm config set registry https://your-internal-npm-registry/
```

Do not paste comment lines like `# paste generated id...` into `zsh`; depending on shell options, that may execute as a command. Paste only the commands.
