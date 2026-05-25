import assert from "node:assert/strict";

import nextConfig from "../next.config.mjs";

const rewrites = await nextConfig.rewrites();

assert.deepEqual(rewrites, [
  {
    source: "/api/:path*",
    destination: "http://backend:8000/api/:path*",
  },
  {
    source: "/static/:path*",
    destination: "http://backend:8000/static/:path*",
  },
]);
