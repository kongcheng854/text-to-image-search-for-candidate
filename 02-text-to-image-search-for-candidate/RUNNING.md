# 运行说明

本项目使用 CLIP ViT-B/32 将图片和英文描述编码到同一个 512 维向量空间。默认在余弦相似度上加入保守的文本侧重排，返回前 10 张图片。所有模型推理均在 CPU 上完成，不需要显卡或付费 API。
使用 NumPy 实现本地精确向量索引：保存图片向量、图片 ID 和路径，并全量计算相似度。无需另外启动数据库服务。向量矩阵约 1.95 MiB。

## 1. 安装环境

建议使用 64 位 Python 3.12。下面命令在项目根目录执行，Windows PowerShell 示例：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch 单独从官方 CPU 源安装，避免安装 GPU 运行库。需要与本次验证完全一致的依赖时，在安装 CPU PyTorch 后把最后一行替换为 `pip install -r requirements-lock.txt`（包含测试依赖）。该锁定清单针对本次 Windows / Python 3.12 环境，其他平台优先使用 `requirements.txt`。

如果使用 uv，可以用 `uv venv --python 3.12` 创建环境，再使用 `uv pip install --python .venv/Scripts/python.exe` 执行相同安装步骤。

## 2. 准备模型

```powershell
.venv\Scripts\python.exe -m src.prepare_model
```

该命令从 OpenAI 官方地址下载约 338 MiB 的原始模型，校验 SHA-256，并转换为本地 Transformers 模型和分词器。文件保存在 `.cache/` 下；转换后的模型使用 float32、safetensors。转换不包含训练或微调。

## 3. 检查图库、建立索引

```powershell
.venv\Scripts\python.exe -m src.dataset
.venv\Scripts\python.exe -m src.build_index --offline
```

默认每批 8 张、最多使用 4 个 CPU 推理线程。索引只读取 `materials/data/images.jsonl` 所列且位于 `materials/data/images/` 内的图片；清单路径相对于 `materials/data/` 解析。遇到重复 ID、越界路径、缺失或损坏图片将中止并说明原因。

```powershell
# 内存较紧张时减小批次；CPU 较慢时可减少线程。
.venv\Scripts\python.exe -m src.build_index --offline --batch-size 2 --threads 2
```

索引位于 `artifacts/index/`，每次成功构建写入一个完整版本，再原子切换 `current.json`。相同图片内容和模型配置下，重复执行会复用已有索引。更换图片后重新建库；必要时使用 `--force` 强制构建。旧版本不会自动删除。

## 4. 打开搜索页面

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器访问 <http://127.0.0.1:8501>。也可以执行 `./start.ps1 -Offline`，或先设置 `$env:FRAMEFIND_OFFLINE='1'` 再启动页面，确保不会尝试联网。

输入英文自然语言，点击“搜索图片”。页面显示按相关性降序排列的 10 张图片、排名、图片 ID 和相关性得分，并可以下载 JSON 结果。首次搜索包括模型加载；页面显示的检索耗时不包括这部分。得分不是概率。展开“检索设置”可切换到基础检索进行对比。

默认优化规则：单一主要对象描述使用相同 CLIP 模型比较目标对象与常见混淆对象，仅在混淆对象分数更高时适度降权；明确包含多个非人物对象时，将原句与 `a photo of ...` 的文本向量归一化平均，不使用互斥类别惩罚。未知对象、否定、选择关系以及 hot dog、bus stop 等易误解复合词保留基础整句检索。词表只包含应用自定义词与同义词，没有查询到图片 ID 的映射。

图片向量索引无需重建；对象文本原型在启动时分批编码并缓存。排序规则版本及词表摘要随结果导出。JSON 中 `score` 是最终排序分数，`clip_score` 保留原句与图片的原始余弦相似度。

也支持命令行：

```powershell
.venv\Scripts\python.exe -m src.search "A dog running on the beach." --offline
# 对比优化前的整句余弦检索：
.venv\Scripts\python.exe -m src.search "A red car." --offline --ranking baseline
```

## 5. 验证与示例

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m src.evaluate --offline
```

评估脚本使用手写英文描述，保存 3 组 Top 10 结果到 `examples/query-*.json`，同时检查结果数量、图片唯一性、路径、降序排列、重复查询一致性。它记录模型加载、查询用时和内存信息到 `examples/verification.json`。它不把这些结构性检查当作语义准确率。

示例查询和截图说明见 `examples/README.md`。程序不下载、不读取 COCO captions、类别标注或原始图文对应关系，也没有预设查询到图片的答案表。

25 条开发查询与新增 10 条验证查询的优化对照、评分口径和局限见 [evaluation/optimization/README.md](evaluation/optimization/README.md)。该评测的判断文件仅供离线汇总，搜索运行时不读取这些文件。

## 复现与限制

- 模型：OpenAI CLIP ViT-B/32。原始 checkpoint SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。
- Hugging Face 路径固定为 `openai/clip-vit-base-patch32` 的 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。转换后的模型和分词器文件也记录内容摘要，加载时校验。
- 图片使用模型配套的 RGB、缩放、中心裁剪、标准化预处理；图片和文本向量均进行 L2 归一化。得分相同时按图片 ID 升序排列。
- 索引保存图片内容摘要、模型 revision、维度和预处理版本，并校验索引文件完整性。图片或模型变化时需重建；程序重启时检查图库内容。
- 固定软件版本、模型与图库后，同机重复查询应一致。跨操作系统、CPU 和数学库可能有很小的浮点差异，极接近的得分可能改变次序。
- CLIP 对小目标、数量、否定、复杂关系和细微颜色可能检索不准，中心裁剪也可能丢失画面边缘的信息。图库不含精确匹配内容时，仍会返回相对最相关的 10 张图片。
- 首次 CPU 建库耗时随机器而变化；建库完成后，查询不再重复编码图片。页面仅监听本机地址。

## 文件说明

| 文件 | 职责 |
| --- | --- |
| `app.py` | 本地搜索页面与模型缓存 |
| `src/dataset.py` | 读取清单、校验图片和生成数据指纹 |
| `src/prepare_model.py` | 官方模型下载、校验和离线转换 |
| `src/encoder.py` | CPU 图片与文本编码、L2 归一化 |
| `src/index.py` | 索引保存、加载和完整性检查 |
| `src/build_index.py` | 分批编码全部图库图片 |
| `src/search.py` | 精确余弦检索和命令行入口 |
| `src/query.py` | 保守对象识别、同义词与回退判断 |
| `src/ranking.py` | 对象混淆降权、多对象文本模板融合 |
| `src/evaluate.py` | 三组查询、重复性检查与性能记录 |


