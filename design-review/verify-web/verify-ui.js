// LuxTTS Studio · 界面冒烟检查
//
// 跑法：node verify-ui.js <outDir> <port>
// 本机没有 playwright 自带浏览器，必须走系统 Chrome：
//   chromium.launch({ headless: true, channel: "chrome" })
//
// 只做「页面能不能起来、三个视图都在不在、有没有 JS 报错」这一层，
// 不碰合成（那要真跑模型，另见 verify-flow.js）。
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || ".";
const PORT = process.argv[3] || "7867";
const URL = `http://127.0.0.1:${PORT}/`;
fs.mkdirSync(OUT, { recursive: true });

const problems = [];
const log = (...a) => console.log(...a);

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    channel: "chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 950, deviceScaleFactor: 1 });

  const consoleErrs = [];
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning")
      consoleErrs.push(`[${m.type()}] ${m.text()}`);
  });
  page.on("pageerror", (e) => consoleErrs.push(`[pageerror] ${e.message}`));
  page.on("requestfailed", (r) =>
    consoleErrs.push(`[requestfailed] ${r.url()} ${r.failure() && r.failure().errorText}`));

  await page.goto(URL, { waitUntil: "networkidle2", timeout: 30000 });
  await page.waitForSelector("#app:not(.hidden)", { timeout: 20000 });

  const check = async (name, fn) => {
    try {
      const r = await fn();
      if (r === true || r === undefined) log(`  [ok]   ${name}`);
      else { log(`  [FAIL] ${name} -> ${r}`); problems.push(`${name}: ${r}`); }
    } catch (e) {
      log(`  [ERR]  ${name} -> ${e.message}`);
      problems.push(`${name}: ${e.message}`);
    }
  };

  log("== 启动 ==");
  await check("左导航恰好三个视图", async () => {
    const n = await page.$$eval(".navitem", (e) => e.length);
    return n === 3 ? true : `实际 ${n} 个`;
  });
  await check("环境芯片已填档位", async () => {
    const t = await page.$eval("#env-main", (e) => e.textContent.trim());
    return t && t !== "读取中…" ? true : `仍是「${t}」`;
  });
  await check("合成页：有生成按钮 / 参数面板 / 结果卡", async () => {
    const r = await page.evaluate(() => ({
      go: !!document.querySelector("#s-go"),
      settings: !!document.querySelector("#s-details details"),
      res: !!document.querySelector("#s-rail .card"),
      chips: !!document.querySelector("#s-voice-chips"),
    }));
    return Object.values(r).every(Boolean) ? true : JSON.stringify(r);
  });
  await check("参数面板四项齐全，且不含音色级参数", async () => {
    const ks = await page.$$eval("#s-details [data-k]", (e) => e.map((x) => x.dataset.k));
    const want = ["num_steps", "speed", "t_shift", "return_smooth"];
    const extra = ks.filter((k) => !want.includes(k));
    const missing = want.filter((k) => !ks.includes(k));
    return (!extra.length && !missing.length) ? true : `缺 ${missing} 多 ${extra}`;
  });
  await check("生成按钮在未选音色时是禁用的", async () =>
    (await page.$eval("#s-go", (e) => e.disabled)) ? true : "居然可点");
  await check("示例按钮能填文本并更新字数", async () => {
    await page.click("#s-sample");
    const n = await page.$eval("#s-count", (e) => e.textContent.trim());
    return /\d+ 字/.test(n) ? true : `字数显示「${n}」`;
  });
  await check("参数面板的提示文案已渲染", async () => {
    const t = await page.$eval("[data-role=num_steps-v]", (e) => e.textContent.trim());
    return t.length > 4 ? true : "提示是空的";
  });

  await page.screenshot({ path: path.join(OUT, "01-synth.png"), fullPage: true });

  log("== 切页 ==");
  for (const [view, file, sel] of [
    ["library", "02-library.png", "#lb-new"],
    ["history", "03-history.png", "#h-list"],
  ]) {
    await check(`切到「${view}」`, async () => {
      await page.click(`.navitem[data-view=${view}]`);
      await page.waitForFunction(
        (v) => !document.querySelector("#view-" + v).classList.contains("hidden"),
        {}, view);
      return true;
    });
    await check(`${view} 关键元素可见`, async () => {
      const ok = await page.$eval(sel, (e) => {
        const r = e.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      });
      return ok ? true : `${sel} 尺寸为 0`;
    });
    await page.screenshot({ path: path.join(OUT, file), fullPage: true });
  }

  log("== 深色主题 ==");
  await check("切深色后令牌与波形重绘不报错", async () => {
    await page.click("#theme-btn");
    await new Promise((r) => setTimeout(r, 400));
    const t = await page.$eval("html", (e) => e.dataset.theme);
    if (t !== "dark") throw new Error(`主题没切过去（${t}）`);
    await page.screenshot({ path: path.join(OUT, "04-dark.png"), fullPage: true });
    await page.click("#theme-btn");
    return true;
  });

  log("== 环境抽屉 ==");
  await check("抽屉能打开并列出档位", async () => {
    await page.click("#env-btn");
    await page.waitForFunction(() =>
      document.querySelector("#env-drawer").classList.contains("on"), { timeout: 3000 });
    const rows = await page.$$eval("#env-body .kv tr", (e) => e.length);
    await page.screenshot({ path: path.join(OUT, "05-env.png") });
    if (rows < 5) throw new Error(`档位只有 ${rows} 行`);
    return true;
  });
  await page.click("#scrim");

  await browser.close();

  log("");
  log("== 控制台 ==");
  const noise = consoleErrs.filter((m) =>
    !/favicon|DevTools|Download the React/i.test(m));
  if (noise.length) {
    noise.forEach((m) => log("  " + m));
    problems.push(`控制台有 ${noise.length} 条报错/警告`);
  } else {
    log("  干净");
  }

  log("");
  if (problems.length) {
    log(`FAILED (${problems.length})`);
    problems.forEach((p) => log("  - " + p));
    process.exit(1);
  }
  log("ALL PASS");
})().catch((e) => { console.error("verify-ui 崩了:", e); process.exit(2); });