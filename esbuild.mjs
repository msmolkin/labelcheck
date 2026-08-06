// Build the labelcheck frontend: static/app.ts -> static/dist/app.js,
// static/app.css -> static/dist/app.css. Both are minified and self-contained.
//
// Two properties this build is responsible for proving, because the app promises
// "nothing leaves the box":
//   1. no sourcemap and no NODE_ENV-conditional branches, so what ships is what was read;
//   2. no absolute URL survives into the bundle - the guard below fails the build if one
//      does, which is the same assertion deploy/deploy.sh repeats before rsync.
import { build, context } from "esbuild";
import { readFileSync } from "node:fs";

const watch = process.argv.includes("--watch");

const options = {
  entryPoints: ["static/app.ts", "static/app.css"],
  outdir: "static/dist",
  bundle: true,
  minify: true,
  sourcemap: false,
  format: "iife",
  target: ["es2020", "chrome109", "firefox115", "safari15"],
  logLevel: "info",
};

const OUTPUTS = ["static/dist/app.js", "static/dist/app.css"];
// Matches http:// and https:// anywhere in the built output. Protocol-relative URLs
// ("//example.com") are caught by the second pattern.
const ABSOLUTE_URL = /https?:\/\//;
const PROTOCOL_RELATIVE = /["'(]\/\/[a-z0-9-]+\./i;

function assertNoRemoteReferences() {
  for (const file of OUTPUTS) {
    const text = readFileSync(file, "utf8");
    for (const pattern of [ABSOLUTE_URL, PROTOCOL_RELATIVE]) {
      const hit = text.match(pattern);
      if (hit) {
        throw new Error(
          `${file} contains a remote reference (${hit[0]}). The app must load nothing ` +
            `but same-origin assets; remove it or the Content-Security-Policy will block it.`,
        );
      }
    }
  }
  console.log(`no remote references in ${OUTPUTS.join(", ")}`);
}

if (watch) {
  const ctx = await context(options);
  await ctx.watch();
  console.log("watching static/app.ts and static/app.css");
} else {
  await build(options);
  assertNoRemoteReferences();
}
