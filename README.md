# Advanced RAG

一个可读、可替换、可验证的本地文档 Advanced RAG 项目。不依赖 LangChain，你能直接跟踪：Document → Chunk → Dense/BM25 → RRF → Reranker → LLM → Answer。

## 支持范围

- Markdown/MDX、HTML、可提取文字的 PDF。
- Markdown 博客会丢弃 `<!-- more -->` 之前的摘要区，避免每日一言、歌词等非正文进入索引与模型上下文。
- FAISS `IndexFlatIP` 的 Dense Retrieval、`jieba` 中文 BM25 与等权 RRF 融合。
- 单次结构化 Query Understanding / Rewrite；百炼 `qwen3-rerank` 二阶段重排。
- 强制基于检索证据回答；无命中时拒答；每次运行留下完整阶段证据。

不包含 OCR、网页抓取、多查询扩展、Agent 循环、本地 CrossEncoder 和前端。

## 安装与配置

需要 Python 3.13：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

在 `.env` 填写 `DASHSCOPE_API_KEY` 与 `DASHSCOPE_WORKSPACE_ID`；后者只在 `hybrid-rerank` 与 `advanced` 使用。密钥不会进入 YAML、索引或运行记录。

所有模型、候选数量、RRF、重排和运行默认值均在 [config/default.yaml](config/default.yaml) 中配置。

## 语料与命令

```powershell
python scripts/snapshot_blog_corpus.py --source "A:\XXY\blog\xxy1103.github.io\src\content\blog"
python -m base_rag ingest --config config/default.yaml
python -m base_rag ask --config config/default.yaml --question "KDD Cup 2026 Data Agents 比赛强调哪些能力？"
python -m base_rag ask --config config/default.yaml --profile dense --question "KDD Cup 2026 Data Agents 比赛强调哪些能力？"
python -m base_rag eval --config config/default.yaml --profile hybrid
python -m base_rag experiment --config config/default.yaml
```

Profile 有 `dense`、`bm25`、`hybrid`、`hybrid-rerank`、`advanced`，默认 `advanced`：问题 → JSON 改写 → Dense/BM25 Top-20 → RRF Top-30 → qwen3-rerank Top-6 → 回答。

升级后需重新执行 `ingest`；旧索引仅可用于 `dense` Profile。建库会输出语料哈希与 BM25 分词器版本。

## 产物与验证

- `data/index/`：FAISS、BM25 词频统计、Chunk 元数据与构建参数。
- `runs/`：改写、各阶段候选与分数、Prompt、回答、引用、调用数和耗时。
- `runs/experiments/`：五组消融的原始结果、报告和人工复核文件。

```powershell
python -m pytest
```

`evaluations/phase2_questions.yaml` 冻结了 40 题分层集。`experiment` 比较五个 Profile 的 Source Recall@6、Chunk Recall@6/@20、MRR@6、nDCG@6、类别指标、延迟和调用数；只为 Dense 与 Advanced 生成最终答案。

实验会显示每个 Profile 的终端进度条，并在每题结束后原子写入 `runs/experiments/<时间戳>/<profile>.json`。单题的模型/审核/网络失败会记录题号、失败阶段和原始错误后继续执行；这些失败题按 0 分计入检索指标，并在汇总中单独可查。

填写实验目录中的 `human_review.csv` 后执行：

```powershell
python -m base_rag experiment-report --experiment runs/experiments/<时间戳> --reviews runs/experiments/<时间戳>/human_review.csv
```

模块是否有效只由消融 Delta 判断：Hybrid 对比 Dense/BM25，Reranker 对比 Hybrid，Rewrite 对比 Hybrid-Rerank。模块成功执行不等于产生效果，正向、中性和负向结果都会保留。
