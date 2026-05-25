import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoFrontend = join(here, "..");

const apiSource = readFileSync(join(repoFrontend, "lib", "api.ts"), "utf8");
const dockerfile = readFileSync(join(repoFrontend, "Dockerfile"), "utf8");
const envExample = readFileSync(join(repoFrontend, ".env.example"), "utf8");

assert.match(apiSource, /function normalizeApiBase/);
assert.doesNotMatch(apiSource, /http:\/\/localhost:8000/);
assert.match(dockerfile, /^ARG NEXT_PUBLIC_API_URL=$/m);
assert.match(envExample, /^NEXT_PUBLIC_API_URL=$/m);
