# 以文搜图搜索引擎

## 目标

输入一段英文自然语言描述，从提供的 1,000 张图片中返回按相关性排序的 Top 10 图片。例如：`An orange cat lying on a sofa.`

## 限制

- 从零实现，技术栈不限，允许 AI；程序须在 CPU、8 GB 内存、无独立显卡的电脑上运行，可依赖第三方 API。
- 检索范围固定为 `materials/data/images/`；图片 ID 和路径见 `materials/data/images.jsonl`。不得使用公开数据集的原始描述或图文对应标注作为检索答案。
- 提交完整源码、简短运行说明，以及 3 组英文查询和对应的检索结果截图。

## 评估标准

- 按说明可启动，输入英文描述后能查看按相关性排序的 10 张不同图库图片。
- 对图库中存在匹配内容的查询，返回结果能体现描述中的主要内容，优先展示更相关的图片。
- 结果可复现，满足设备限制；界面精美程度不作为重点。

图片来自 [COCO](https://cocodataset.org/#download)，版权与使用条款见[来源说明](https://cocodataset.org/#termsofuse)。

---

## 已实现项目

使用 CLIP ViT-B/32 在 CPU 上编码图片与英文描述，基于本地 NumPy 向量索引进行精确检索，并对对象混淆和多对象措辞做保守重排。页面展示按相关性排序的 10 张不同图片，支持切换基础检索对比和下载检索记录。模型准备完成后可以离线运行，不使用 COCO 原始描述或图文对应标注。

### 快速运行（Windows / Python 3.12）

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.prepare_model
.venv\Scripts\python.exe -m src.build_index --offline
.venv\Scripts\python.exe -m streamlit run app.py
```

打开 <http://127.0.0.1:8501>，输入英文描述并搜索。若当前目录已完成环境、模型和索引准备，直接执行最后一行即可，也可以使用 `./start.ps1 -Offline`。

完整安装、设计、复现和限制说明见 [RUNNING.md](RUNNING.md)。三组查询的真实结果截图及验证记录见 [examples/README.md](examples/README.md)。

检索质量优化、25 条开发查询和 10 条新增验证查询的实际对照见 [优化报告](evaluation/optimization/README.md)。
