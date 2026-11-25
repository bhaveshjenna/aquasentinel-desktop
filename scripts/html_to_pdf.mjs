/**
 * Print a local HTML file to PDF via the Chrome DevTools Protocol.
 *
 *   node scripts/html_to_pdf.mjs <input.html> <output.pdf>
 *
 * Why CDP rather than `chrome --print-to-pdf`: that flag stamps Chrome's own header and footer
 * onto every page - a timestamp, the document title, and the full local file:// path - with no
 * supported CLI switch to turn them off (`--no-pdf-header-footer` is a Puppeteer option, not a
 * Chrome one; passing it makes Chrome print a blank page instead). Page.printToPDF takes
 * `displayHeaderFooter: false` directly, and also lets us force background graphics on and set
 * the paper size explicitly rather than inheriting a locale default.
 *
 * Uses Node's global WebSocket (Node 22+) - no dependencies.
 */

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const CANDIDATES = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
];

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: node html_to_pdf.mjs <input.html> <output.pdf>");
  process.exit(2);
}

const browser = CANDIDATES.find((p) => existsSync(p));
if (!browser) {
  console.error("no Chrome or Edge found");
  process.exit(1);
}

const profile = mkdtempSync(join(tmpdir(), "cdp-pdf-"));
const port = 9200 + Math.floor(Math.random() * 400);

const child = spawn(browser, [
  "--headless",
  "--disable-gpu",
  "--no-sandbox",
  "--no-first-run",
  "--disable-extensions",
  `--user-data-dir=${profile}`,
  `--remote-debugging-port=${port}`,
  "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function endpoint() {
  // Attach to a PAGE target, not the browser target: the browser endpoint has no Page
  // domain, so Page.enable there fails with "wasn't found".
  for (let i = 0; i < 80; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/list`);
      if (res.ok) {
        const page = (await res.json()).find(
          (t) => t.type === "page" && t.webSocketDebuggerUrl,
        );
        if (page) return page.webSocketDebuggerUrl;
      }
    } catch { /* not up yet */ }
    await sleep(150);
  }
  throw new Error("Chrome did not expose a page debugging target");
}

function connect(url) {
  return new Promise((ok, bad) => {
    const ws = new WebSocket(url);
    ws.onopen = () => ok(ws);
    ws.onerror = (e) => bad(new Error(`websocket failed: ${e?.message ?? e}`));
  });
}

function rpc(ws) {
  let next = 1;
  const pending = new Map();
  const events = new Map();
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      const { ok, bad } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? bad(new Error(JSON.stringify(msg.error))) : ok(msg.result);
    } else if (msg.method && events.has(msg.method)) {
      events.get(msg.method)();
      events.delete(msg.method);
    }
  };
  return {
    send(method, params = {}) {
      const id = next++;
      return new Promise((ok, bad) => {
        pending.set(id, { ok, bad });
        ws.send(JSON.stringify({ id, method, params }));
      });
    },
    once(method) {
      return new Promise((ok) => events.set(method, ok));
    },
  };
}

try {
  const ws = await connect(await endpoint());
  const cdp = rpc(ws);

  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  const target = pathToFileURL(resolve(input)).href;
  const loaded = cdp.once("Page.loadEventFired");
  const nav = await cdp.send("Page.navigate", { url: target });
  if (nav.errorText) throw new Error(`navigate failed: ${nav.errorText}`);
  await Promise.race([loaded, sleep(30000)]);
  // Let webfonts settle and any layout-dependent SVG text metrics resolve.
  await sleep(1500);

  // Verify we are actually about to print the document rather than about:blank - the failure
  // mode that silently yields a one-page 23 KB PDF.
  const probe = await cdp.send("Runtime.evaluate", {
    expression: "JSON.stringify({url: location.href, h: document.body.scrollHeight})",
    returnByValue: true,
  });
  const info = JSON.parse(probe.result.value);
  if (!info.url.startsWith("file:") || info.h < 2000) {
    throw new Error(`page did not load: ${info.url} (height ${info.h})`);
  }

  // ReturnAsStream: a large document returned inline is one multi-megabyte websocket frame,
  // and above a few MB that response never arrives - the print simply hangs. Streaming it
  // back in 1 MB reads works at any size.
  const { stream } = await cdp.send("Page.printToPDF", {
    printBackground: true,
    displayHeaderFooter: false,   // the whole reason for using CDP
    preferCSSPageSize: true,      // honour @page { size: A4; margin: ... }
    paperWidth: 8.27,
    paperHeight: 11.69,
    marginTop: 0.67,
    marginBottom: 0.71,
    marginLeft: 0.63,
    marginRight: 0.63,
    scale: 1,
    transferMode: "ReturnAsStream",
  });
  const chunks = [];
  for (;;) {
    const r = await cdp.send("IO.read", { handle: stream, size: 1 << 20 });
    chunks.push(Buffer.from(r.data, r.base64Encoded ? "base64" : "binary"));
    if (r.eof) break;
  }
  await cdp.send("IO.close", { handle: stream });

  writeFileSync(resolve(output), Buffer.concat(chunks));
  ws.close();
  console.log(`wrote ${output}`);
} catch (err) {
  console.error(String(err?.message ?? err));
  process.exitCode = 1;
} finally {
  child.kill();
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
}
