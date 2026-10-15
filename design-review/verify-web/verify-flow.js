// LuxTTS Studio · 端到端真实流程检查
//
// 跑法：node verify-flow.js <outDir> <port> <参考音频绝对路径>
//
// 这一版会**真的跑模型**：上传参考音频 → 试转写 → 保存音色 → 合成 → 看记录 → 删除音色。
// 冷启动要十几秒（模型装载 + CUDA 预热；转写用 SenseVoice，冷启只要几秒），
// 所以每个等待都给了宽裕的超时。
//
// 收尾一定把测试音色删掉 —— 参考音频取自别人（OmniVoice 的音色库），
// 留着就是个不请自来的克隆。
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || ".";
const PORT = process.argv[3] || "7867";
const REF = process.argv[4];
const URL = `http://127.0.0.1:${PORT}/`;
if (!REF || !fs.existsSync(REF)) {
  console.error("用法：node verify-flow.js <outDir> <port> <参考音频路径>");
  process.exit(2);
}
fs.mkdirSync(OUT, { recursive: true });

const problems = [];
const created = [];
const log = (...a) => console.log(...a);
const t0 = Date.now();
const el = () => `${((Date.now() - t0) / 1000).toFixed(0)}s`;

async function step(name, fn) {
  const s = Date.now();
  try {
    const r = await fn();
    log(`  [ok]   ${name} (${((Date.now() - s) / 1000).toFixed(1)}s)`
        + (typeof r === "string" && r ? ` · ${r}` : ""));
    return r;
  } catch (e) {
    log(`  [FAIL] ${name} -> ${e.message}`);
    problems.push(`${name}: ${e.message}`);
    return null;
  }
}
async function until(page, fn, msg, timeout = 180000, ...args) {
  // args 透传给页面内的回调：回调跑在浏览器上下文，闭包变量（Node 侧的值）它看不见。
  await page.waitForFunction(fn, { timeout, polling: 400 }, ...args);
  void msg;
}

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    channel: "chrome",
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 950 });
  const errs = [];
  page.on("pageerror", (e) => errs.push(`[pageerror] ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") errs.push(`[console] ${m.text()}`); });

  await page.goto(URL, { waitUntil: "networkidle2" });
  await page.waitForSelector("#app:not(.hidden)");

  log(`== 新建音色 ==（${el()}）`);
  await page.click('.navitem[data-view="library"]');
  await page.click("#lb-new");
  await page.waitForSelector("#dlg-new[open]");

  await step("上传参考音频后解码出波形", async () => {
    const input = await page.$("#n-drop input[type=file]");
    await input.uploadFile(REF);
    await until(page, () => {
      const b = document.querySelector("#n-badge");
      return b && (b.dataset.state === "done" || b.dataset.state === "error");
    }, "解码");
    const st = await page.$eval("#n-badge", (e) => e.dataset.state);
    if (st !== "done") throw new Error("解码没成功");
    const note = await page.$eval("#n-editor .ed-note", (e) => e.textContent.trim());
    if (!note) throw new Error("裁剪提示是空的");
    return note;
  });

  await step("试转写（真跑 SenseVoice）", async () => {
    await page.click("#n-asr");
    await until(page, () => {
      const b = document.querySelector("#n-badge");
      return b && (b.dataset.state === "done" || b.dataset.state === "error");
    }, "转写", 300000);
    const st = await page.$eval("#n-badge", (e) => e.dataset.state);
    if (st !== "done") {
      const msg = await page.$eval("#n-alerts", (e) => e.textContent.trim());
      throw new Error("转写失败：" + msg);
    }
    const txt = await page.$eval("#n-text", (e) => e.value.trim());
    if (!txt) throw new Error("转写结果为空");
    return `“${txt.slice(0, 40)}…”`;
  });

  await step("命名并保存（真建 prompt 缓存）", async () => {
    await page.$eval("#n-name", (e) => { e.value = ""; });
    await page.type("#n-name", "验证用音色");
    await page.click("#n-save");
    await until(page, () => !document.querySelector("#dlg-new").open, "保存", 300000);
    await until(page, () =>
      document.querySelectorAll("#lb-list .vitem").length > 0, "列表出现", 30000);
    const n = await page.$eval("#lb-name", (e) => e.textContent.trim());
    created.push(n);
    return `已保存「${n}」`;
  });

  await step("详情面板显示转写文本与参考时长", async () => {
    const v = await page.evaluate(() => ({
      text: document.querySelector("#lb-reftext").value.trim(),
      dur: document.querySelector("#lb-refdur").value,
      rms: document.querySelector("#lb-rms").value,
    }));
    if (!v.text) throw new Error("参考文本框是空的");
    if (!v.dur || !v.rms) throw new Error(`参考时长/响度没回填：${JSON.stringify(v)}`);
    return `ref_duration=${v.dur} rms=${v.rms}`;
  });

  await page.screenshot({ path: path.join(OUT, "10-library.png"), fullPage: true });

  log(`== 合成 ==（${el()}）`);
  await step("选音色 + 填文本 + 生成（真跑生成）", async () => {
    await page.click('.navitem[data-view="synth"]');
    await until(page, () => document.querySelectorAll("#s-voice-chips .vchip").length > 0,
      "芯片出现", 15000);
    await page.click("#s-voice-chips .vchip");
    await page.$eval("#s-text", (e) => { e.value = ""; });
    await page.type("#s-text", "这是一次端到端验证，数字一二三四五。");
    const dis = await page.$eval("#s-go", (e) => e.disabled);
    if (dis) throw new Error("选完音色后生成按钮仍是禁用的");
    await page.click("#s-go");

    // 等待时不能只死等：poll 一旦失败（`Waiting failed` 多半是页面上下文没了，
    // 而不是超时），要把当时页面上的提示文案和控制台错误一起带出来，否则只剩
    // 三个字，什么都定位不了。
    try {
      await until(page, () => {
        const b = document.querySelector("#s-rail .pill");
        return b && (b.dataset.state === "done" || b.dataset.state === "error");
      }, "生成结束", 420000);
    } catch (e) {
      const diag = await page.evaluate(() => {
        const pill = document.querySelector("#s-rail .pill");
        const notes = document.querySelector("#s-rail .notes");
        return {
          pillState: pill ? pill.dataset.state : "(没有 pill)",
          pillText: pill ? pill.textContent.trim() : "",
          notes: notes ? notes.textContent.trim() : "(没有提示条)",
        };
      }).catch((x) => ({ evalError: x.message }));
      throw new Error(`${e.message} | 诊断：pill=${diag.pillState} `
        + `“${diag.pillText}” 提示=${diag.notes}`
        + (diag.evalError ? ` 取诊断本身也失败了：${diag.evalError}` : ""));
    }
    const st = await page.$eval("#s-rail .pill", (e) => e.dataset.state);
    if (st !== "done") {
      const msg = await page.$eval("#s-rail .notes", (e) => e.textContent.trim());
      throw new Error("生成失败：" + msg);
    }
    const m = await page.$$eval("#s-rail .metric", (es) => es.map((e) => e.textContent.trim()));
    return m.join(" · ");
  });

  await step("结果卡回显了四类统计与生成参数", async () => {
    const r = await page.evaluate(() => ({
      metrics: document.querySelectorAll("#s-rail .metric").length,
      params: document.querySelectorAll("#s-rail .pmeta__c").length,
      src: document.querySelector("#s-rail .wpl") ? "有播放器" : "无播放器",
    }));
    if (r.metrics < 4) throw new Error(`统计只有 ${r.metrics} 项`);
    if (r.params < 5) throw new Error(`参数只有 ${r.params} 项`);
    return `${r.metrics} 项统计 / ${r.params} 项参数 / ${r.src}`;
  });

  await step("采样率按 48 kHz 呈现", async () => {
    const t = await page.$$eval("#s-rail .metric", (es) =>
      es.map((e) => e.textContent).join("|"));
    if (!/48 kHz/.test(t)) throw new Error(`没看到 48 kHz：${t}`);
    return true;
  });

  await page.screenshot({ path: path.join(OUT, "11-synth-done.png"), fullPage: true });

  log(`== 合成记录 ==（${el()}）`);
  await step("记录页出现这一条，且带试听与重新生成", async () => {
    await page.click('.navitem[data-view="history"]');
    await until(page, () => document.querySelectorAll("#h-list .hrow").length > 0, "记录出现");
    const r = await page.evaluate(() => {
      const row = document.querySelector("#h-list .hrow");
      return {
        text: row.querySelector(".hrow__text").textContent.trim(),
        meta: Array.from(row.querySelectorAll(".hrow__meta span")).map((e) => e.textContent.trim()),
        acts: Array.from(row.querySelectorAll(".hrow__acts button")).map((e) => e.textContent.trim()),
      };
    });
    if (!/端到端验证/.test(r.text)) throw new Error("记录正文不对：" + r.text);
    if (!r.acts.includes("试听") || !r.acts.includes("重新生成")) {
      throw new Error("操作行缺按钮：" + JSON.stringify(r.acts));
    }
    return r.meta.join(" · ");
  });

  await step("点「重新生成」把输入回填到合成页", async () => {
    await page.click('#h-list .hrow [data-role=reuse]');
    await until(page, () =>
      !document.querySelector("#view-synth").classList.contains("hidden"), "切回合成页");
    const t = await page.$eval("#s-text", (e) => e.value.trim());
    const on = await page.$$eval("#s-voice-chips .vchip.is-on", (e) => e.length);
    const steps = await page.$eval("#f-steps", (e) => e.value);
    if (!/端到端验证/.test(t)) throw new Error("文本没回填");
    if (on !== 1) throw new Error("音色没回填（选中数 " + on + "）");
    return `文本已回填 · 音色已选中 · 步数回填为 ${steps}`;
  });

  await page.screenshot({ path: path.join(OUT, "12-history-echo.png"), fullPage: true });

  log(`== 清理 ==（${el()}）`);
  await step("删除测试音色", async () => {
    // 注意：waitForFunction 的回调跑在**页面上下文**里，引用不到 Node 侧的
    // `created` —— 早期版本就这么写，结果报 pageerror「created is not defined」。
    // 名字必须按参数传进去。
    const name = created[0] || "验证用音色";
    await page.click('.navitem[data-view="library"]');
    await until(page, (n) =>
      Array.from(document.querySelectorAll(".vitem__name")).some(
        (e) => e.textContent.trim() === n), "找到测试音色", 30000, name);
    await page.evaluate((n) => {
      const item = Array.from(document.querySelectorAll(".vitem"))
        .find((d) => d.querySelector(".vitem__name").textContent.trim() === n);
      item.click();
    }, name);
    await page.waitForSelector("#lb-detail:not(.hidden)");
    await page.click("#lb-del");
    await page.waitForSelector("#dlg-confirm[open]");
    await page.click("#cf-ok");
    await until(page, (n) =>
      !Array.from(document.querySelectorAll(".vitem__name")).some(
        (e) => e.textContent.trim() === n), "音色已消失", 30000, name);
    return `已从音色库移除「${name}」`;
  });

  await browser.close();

  log("");
  if (errs.length) {
    log("== 控制台 ==");
    errs.forEach((e) => log("  " + e));
    problems.push(`控制台 ${errs.length} 条报错`);
  }
  if (problems.length) {
    log(`FAILED (${problems.length})`);
    problems.forEach((p) => log("  - " + p));
    process.exit(1);
  }
  log(`ALL PASS（总耗时 ${el()}）`);
})().catch((e) => { console.error("verify-flow 崩了:", e); process.exit(2); });