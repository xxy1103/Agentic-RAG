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
python -m base_rag eval --config config/default.yaml
```

Profile 有 `dense`、`bm25`、`hybrid`、`hybrid-rerank`、`advanced`，默认 `advanced`：问题 → JSON 改写 → Dense/BM25 Top-20 → RRF Top-30 → qwen3-rerank Top-6 → 回答。`ask` 可用 `--profile` 选择其中之一。

升级后需重新执行 `ingest`；旧索引仅可用于 `dense` Profile。建库会输出语料哈希与 BM25 分词器版本。

## 产物与验证

- `data/index/`：FAISS、BM25 词频统计、Chunk 元数据与构建参数。
- `runs/<profile>/`：单题运行记录，按 `dense`、`bm25`、`hybrid`、`hybrid-rerank`、`advanced` 分目录保存改写、各阶段候选与分数、Prompt、回答、引用、调用数和耗时。

```powershell
python -m pytest
```

`evaluations/phase2_questions.yaml` 冻结了 60 题全量分层集，覆盖精确词项、语义改写、多证据组合、模糊表达和无答案问题。`eval` 是唯一评测命令：一次运行全部五种 Profile，且只评测 RAG 检索链路，不调用 LLM 生成最终答案。它汇总 Source Recall@6、Chunk Recall@6/@20、Evidence Coverage@6/@20、MRR@6、nDCG@6、类别指标、延迟和调用数。所有检索指标均适配多文档题：Source Recall 按必需来源覆盖比例计算，Chunk Recall 按唯一 Gold Chunk 覆盖比例计算，Evidence Coverage 按必需事实锚点覆盖比例计算，MRR 对每个必需 Gold Chunk 的倒数排名取平均，nDCG 的理想排序包含全部必需 Gold Chunk。这样只命中部分文档不会被视为完整成功。无答案题仅汇总无答案误检率；拒答成功率和误拒率属于端到端生成评测，不在本命令中统计。

每次 `eval` 都会在 `runs/retrieval-evaluations/<时间戳>/` 写入一次完整的消融批次：根目录的 `summary.json` 与 `REPORT.md` 横向比较全部 Profile，`profiles/<profile>/` 下保存每个 Profile 的 `summary.json`、`REPORT.md` 和 `questions/` 下全部 60 道题的检索运行记录。

评测默认在同一个 Profile 内并发执行 4 道题；并发数可通过 `evaluation.concurrency` 调整，设为 `1` 即恢复串行。并发模式显示线程安全的题目总进度，串行模式额外显示当前题目的阶段进度。FAISS/BM25 索引在每个 Profile 内只加载一次，结果始终按题集原顺序保存。

Embedding、生成和 Rerank 请求遇到连接失败、超时、HTTP 408/409/425/429 或 5xx 时，会按照 `models.max_retries` 自动重试，并在每次重试前固定等待 `models.retry_delay_seconds`（默认 5 秒）。鉴权失败、参数错误等不可恢复请求立即失败。单题最终失败后仍会记录题号、失败阶段和原始错误并继续评测；失败题按 0 分计入检索指标，并在汇总中单独可查。
