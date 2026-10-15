"""Run: python -m streamlit run app.py"""

import json
import os
import time
from dataclasses import asdict

import streamlit as st
from PIL import Image, ImageOps

from src.config import INDEX_DIR, ROOT
from src.search import SearchEngine
from src.ranking import ranking_signature

st.set_page_config(page_title="Framefind · 以文搜图", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 4.5rem; padding-bottom: 1rem;}
h1 {letter-spacing: -0.04em; font-size: 2.6rem !important;}
.eyebrow {color: #247A66; font-size: .8rem; font-weight: 700; letter-spacing: .15em;}
div[data-testid="stImage"] img {border-radius: 9px;}
div[data-testid="stCaptionContainer"] {font-size: .8rem;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(max_entries=1, show_spinner=False)
def get_engine(index_revision: str, offline: bool, ranking: str, ranking_revision: str):
    return SearchEngine(offline=offline, ranking=ranking)


@st.cache_data(max_entries=1000, show_spinner=False)
def thumbnail(path: str, modified: int):
    with Image.open(path) as image:
        # Contain preserves the whole scene, including objects near the edges.
        return ImageOps.pad(image.convert("RGB"), (400, 250), color="#EDF0E9")


st.markdown('<div class="eyebrow">FRAMEFIND / LOCAL IMAGE SEARCH</div>', unsafe_allow_html=True)
st.title("用一句话，找到那一帧。")
st.write("输入英文描述，从 1,000 张图库图片中发现最相关的 10 张。")

default_query = st.query_params.get("q", "")
if "query_input" not in st.session_state:
    st.session_state.query_input = default_query
if "baseline_input" not in st.session_state:
    st.session_state.baseline_input = st.query_params.get("ranking") == "baseline"
with st.form("search_form"):
    columns = st.columns([6, 1], vertical_alignment="bottom")
    query = columns[0].text_input(
        "英文图片描述", key="query_input",
        placeholder="e.g. A dog running on the beach.", max_chars=1000,
    )
    submitted = columns[1].form_submit_button("搜索图片", type="primary", width="stretch")
    with st.expander("检索设置"):
        use_baseline = st.checkbox("使用基础检索进行对比", key="baseline_input",
                                   help="关闭时使用优化检索；开启后使用原始图片与整句描述的相似度排序。")

pointer = INDEX_DIR / "current.json"
if not pointer.exists():
    st.info("首次使用请先建立图片索引，完成后刷新页面。")
    st.code("python -m src.build_index", language="bash")
    st.stop()

# Query parameters make results reproducible via a local URL and facilitate screenshots.
auto_query = bool(default_query) and "results" not in st.session_state
if submitted or auto_query:
    if not query.strip():
        st.warning("请输入一段英文图片描述。")
    else:
        try:
            offline = os.environ.get("FRAMEFIND_OFFLINE", "0") == "1"
            ranking = "baseline" if use_baseline else "optimized"
            with st.spinner("正在理解描述并检索图库…"):
                engine = get_engine(pointer.read_text(encoding="utf-8"), offline, ranking,
                                    json.dumps(ranking_signature(), sort_keys=True))
                started = time.perf_counter()
                results = engine.search(query)
                elapsed = time.perf_counter() - started
            st.session_state.results = results
            st.session_state.result_query = query.strip()
            st.session_state.elapsed = elapsed
            st.session_state.ranking = ranking
            st.session_state.ranking_metadata = engine.ranking_metadata
            st.session_state.model_metadata = engine.metadata["signature"]
            st.session_state.dataset_sha256 = engine.metadata["dataset_sha256"]
            st.query_params["q"] = query.strip()
            st.query_params["ranking"] = ranking
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f"搜索暂时无法完成：{exc}")
            st.stop()

if "results" in st.session_state:
    results = st.session_state.results
    st.subheader(st.session_state.result_query)
    st.caption(
        f"TOP 10 · {'优化检索' if st.session_state.get('ranking', 'optimized') == 'optimized' else '基础检索'}"
        f" · 检索用时 {st.session_state.elapsed:.2f} 秒（不含首次模型加载）"
    )
    for start in (0, 5):
        columns = st.columns(5, gap="medium")
        for column, result in zip(columns, results[start:start + 5]):
            absolute = ROOT / result.path
            with column:
                st.image(thumbnail(str(absolute), absolute.stat().st_mtime_ns), width="stretch")
                st.markdown(f"**{result.rank:02d}** · 相关性 **{result.score:.4f}**")
                st.caption(result.image_id)
    payload = {
        "query": st.session_state.result_query,
        "ranking": st.session_state.get("ranking_metadata"),
        "model": st.session_state.get("model_metadata"),
        "dataset_sha256": st.session_state.get("dataset_sha256"),
        "results": [asdict(result) for result in results],
    }
    st.download_button(
        "下载本次检索记录", data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="search-results.json", mime="application/json",
    )
    st.caption("得分用于比较相关性，不代表匹配概率。图片来自题目提供的 COCO 图库。")
else:
    st.caption("可以描述主体、动作、颜色或场景。首次搜索需要加载模型，请稍候。")

