/** Optional browser verification: install Playwright + Edge, then run this file.
 * The Python app must already be running at http://127.0.0.1:8501.
 */
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');

async function waitForResults(page, expectedIds) {
  // Streamlit sends images and captions progressively; wait for the complete
  // result, rather than treating its first rendered image as the entire grid.
  await page.waitForFunction(expected => {
    const images = [...document.querySelectorAll('[data-testid="stImage"] img')];
    const ids = [...document.querySelectorAll('[data-testid="stCaptionContainer"]')]
      .map(node => node.textContent.trim()).filter(text => text.startsWith('coco-'));
    return images.length === 10 && images.every(image => image.complete && image.naturalWidth > 0)
      && JSON.stringify(ids) === JSON.stringify(expected);
  }, expectedIds, { timeout: 180000 });
}

async function main() {
  const root = path.resolve(__dirname, '..');
  const cases = [1, 2, 3].map(number => JSON.parse(
    fs.readFileSync(path.join(root, 'examples', `query-${number}.json`), 'utf8')
  ));
  const queries = cases.map(item => item.query);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1040 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    for (const [index, query] of queries.entries()) {
      await page.goto(`http://127.0.0.1:8501/?q=${encodeURIComponent(query)}`);
      await page.getByText('TOP 10', { exact: false }).waitFor({ timeout: 180000 });
      await waitForResults(page, cases[index].results.map(row => row.image_id));
      const images = page.locator('[data-testid="stImage"] img');
      if (await images.count() !== 10) throw new Error('Expected ten images');
      await images.evaluateAll(nodes => Promise.all(nodes.map(image => image.decode())));
      const captions = await page.locator('[data-testid="stCaptionContainer"]').allTextContents();
      const displayedIds = captions.map(text => text.trim()).filter(text => text.startsWith('coco-'));
      if (JSON.stringify(displayedIds) !== JSON.stringify(cases[index].results.map(row => row.image_id))) {
        throw new Error('Displayed image IDs do not match the saved search result order');
      }
      await page.screenshot({ path: path.join(root, 'examples', `query-${index + 1}.png`), fullPage: true });
      console.log(`Saved query-${index + 1}.png: ${query}`);
    }
    // Check both ranking modes through the actual UI and compare with recorded results.
    const auditRoot = path.join(root, 'evaluation');
    const original = JSON.parse(fs.readFileSync(path.join(auditRoot, 'baseline.json'), 'utf8'));
    const selected = JSON.parse(fs.readFileSync(path.join(auditRoot, 'optimization', 'development-final.json'), 'utf8'));
    const baselineCar = original.queries.find(item => item.id === 'C05');
    const optimizedCar = selected.find(item => item.id === 'C05');
    await page.goto('http://127.0.0.1:8501/?q=A%20red%20car.&ranking=baseline');
    await page.getByText('TOP 10 · 基础检索', { exact: false }).waitFor({ timeout: 180000 });
    await waitForResults(page, baselineCar.results.map(row => row.image_id));
    await page.locator('[data-testid="stImage"] img').evaluateAll(nodes => Promise.all(nodes.map(image => image.decode())));
    let ids = (await page.locator('[data-testid="stCaptionContainer"]').allTextContents()).map(text => text.trim()).filter(text => text.startsWith('coco-'));
    if (JSON.stringify(ids) !== JSON.stringify(baselineCar.results.map(row => row.image_id))) throw new Error('Baseline comparison mismatch');
    await page.screenshot({ path: path.join(auditRoot, 'optimization', 'red-car-baseline.png'), fullPage: true });
    await page.getByText('检索设置', { exact: true }).click();
    // Streamlit styles the native checkbox input as hidden; users click its label.
    await page.getByText('使用基础检索进行对比', { exact: true }).click();
    if (await page.getByRole('checkbox', { name: '使用基础检索进行对比' }).isChecked()) {
      throw new Error('Ranking mode checkbox did not change');
    }
    await page.getByRole('button', { name: '搜索图片', exact: true }).click();
    await page.getByText('TOP 10 · 优化检索', { exact: false }).waitFor({ timeout: 180000 });
    await waitForResults(page, optimizedCar.results.map(row => row.image_id));
    await page.locator('[data-testid="stImage"] img').evaluateAll(nodes => Promise.all(nodes.map(image => image.decode())));
    ids = (await page.locator('[data-testid="stCaptionContainer"]').allTextContents()).map(text => text.trim()).filter(text => text.startsWith('coco-'));
    if (JSON.stringify(ids) !== JSON.stringify(optimizedCar.results.map(row => row.image_id))) throw new Error('Optimized comparison mismatch');
    await page.getByText('检索设置', { exact: true }).click();
    await page.getByText('使用基础检索进行对比', { exact: true }).waitFor({ state: 'hidden' });
    await page.screenshot({ path: path.join(auditRoot, 'optimization', 'red-car-optimized.png'), fullPage: true });
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: '下载本次检索记录', exact: true }).click();
    const download = await downloadPromise;
    const record = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    if (record.results.length !== 10 || !record.ranking?.version || record.results.some(row => !Number.isFinite(row.clip_score))) {
      throw new Error('Downloaded result lacks ranking metadata or original cosine scores');
    }
    // Exercise the form, rather than checking only query-parameter links.
    await page.getByRole('textbox', { name: '英文图片描述' }).fill(queries[0]);
    await page.getByRole('button', { name: '搜索图片', exact: true }).click();
    await page.getByRole('heading', { name: queries[0], exact: true }).waitFor();
    await page.getByRole('textbox', { name: '英文图片描述' }).fill('');
    await page.getByRole('button', { name: '搜索图片', exact: true }).click();
    await page.getByText('请输入一段英文图片描述。', { exact: true }).waitFor();
    await page.getByRole('textbox', { name: '英文图片描述' }).fill('A dog '.repeat(50));
    await page.getByRole('button', { name: '搜索图片', exact: true }).click();
    await page.getByText('Description is too long', { exact: false }).waitFor();
    if (errors.length) throw new Error(errors.join('\n'));
    fs.writeFileSync(path.join(root, 'examples', 'browser-verification.json'), JSON.stringify({
      passed: true,
      browser: 'Microsoft Edge / Playwright, headless',
      viewport: { width: 1440, height: 1040 },
      checks: [
        '3 query pages render 10 decoded images each',
        'Displayed image IDs and order match saved JSON results',
        'Baseline and optimized modes match audit records, including UI mode switching',
        'Downloaded JSON includes ranking version and original CLIP scores',
        'Form submission updates results',
        'Empty input displays a warning',
        'Overlong input displays an actionable error',
        'No uncaught browser page errors',
      ],
    }, null, 2) + '\n');
    console.log('Browser checks passed: 3 result pages, 10 images each, matching result IDs, form submission, empty/overlong input handling, no page errors.');
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });

