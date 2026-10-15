// 直接在页面上下文里复现 POST /api/jobs，看 fetch 到底卡在哪一步
// 跑法：node probe-fetch.js <port>
const puppeteer = require("puppeteer-core");
const PORT = process.argv[2] || "7867";

(async () => {
  const browser = await puppeteer.launch({
    headless: true, channel: "chrome", args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.log("  [pageerror]", e.message));
  page.on("console", (m) => console.log(`  [${m.type()}]`, m.text()));
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "domcontentloaded" });

  console.log("A) 纯 fetch POST + 读 body：");
  const a = await page.evaluate(async () => {
    const t0 = Date.now();
    try {
      const r = await fetch("/api/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "抓包测试。", voice: "验证用音色",
          params: { num_steps: 4, speed: 0.8, t_shift: 0.9, return_smooth: false },
        }),
      });
      const body = await r.text();
      return { ms: Date.now() - t0, status: r.status, ct: r.headers.get("content-type"), body };
    } catch (e) { return { ms: Date.now() - t0, err: String(e) }; }
  });
  console.log("   ", JSON.stringify(a));

  const id = (() => { try { return JSON.parse(a.body).job.id; } catch (_) { return null; } })();
  console.log("job.id =", id);

  console.log("B) 轮询那个 job：");
  const b = await page.evaluate(async (jid) => {
    const out = [];
    for (let i = 0; i < 20; i++) {
      const t0 = Date.now();
      const r = await fetch("/api/jobs/" + jid);
      const j = await r.json();
      out.push(`${i * 0.6}s ${j.state}/${j.progress}% ${(j.stage || "")} · ${Date.now() - t0}ms`);
      if (j.state === "done" || j.state === "error") break;
      await new Promise((r2) => setTimeout(r2, 600));
    }
    return out;
  }, id);
  b.forEach((l) => console.log("   ", l));

  console.log("C) 页面里的 #s-rail 徽章 + 现存 chip 数：");
  console.log("   ", JSON.stringify(await page.evaluate(() => ({
    pills: Array.from(document.querySelectorAll("#s-rail .pill")).map((p) => `${p.dataset.state}:${p.textContent.trim()}`),
    chips: document.querySelectorAll("#s-voice-chips .vchip").length,
  }))));

  await browser.close();
})().catch((e) => { console.error(e); process.exit(2); });