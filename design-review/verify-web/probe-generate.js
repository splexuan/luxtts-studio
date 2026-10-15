// 盯住合成结果卡的徽章状态，定位「生成完成了但界面没换状态」的问题
// 跑法：node probe-generate.js <port>
const puppeteer = require("puppeteer-core");
const PORT = process.argv[2] || "7867";

(async () => {
  const browser = await puppeteer.launch({
    headless: true, channel: "chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 950 });
  page.on("pageerror", (e) => console.log("  [pageerror]", e.message));
  page.on("console", (m) => { if (m.type() === "error") console.log("  [console]", m.text()); });

  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "networkidle2" });
  await page.waitForSelector("#app:not(.hidden)");

  console.log("chips:", await page.$$eval("#s-voice-chips .vchip", (e) => e.length));
  await page.click("#s-voice-chips .vchip");
  await page.$eval("#s-text", (e) => { e.value = ""; });
  await page.type("#s-text", "第二次验证，短一点。");
  console.log("go disabled:", await page.$eval("#s-go", (e) => e.disabled));

  await page.click("#s-go");
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 500));
    const st = await page.evaluate(() => {
      const pills = Array.from(document.querySelectorAll("#s-rail .pill"));
      return {
        count: pills.length,
        first: pills[0] ? `${pills[0].textContent.trim()}|${pills[0].dataset.state}` : null,
        all: pills.map((p) => `${p.dataset.state}:${p.textContent.trim()}`),
        metrics: document.querySelectorAll("#s-rail .metric").length,
        msg: (document.querySelector("#s-rail .run__msg") || {}).textContent,
      };
    });
    console.log(`  ${(i + 1) * 0.5}s`, JSON.stringify(st));
    if (st.first && (st.first.endsWith("|done") || st.first.endsWith("|error"))) break;
  }
  await page.screenshot({ path: "design-review/verify-web/shots/20-probe-generate.png", fullPage: true });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(2); });