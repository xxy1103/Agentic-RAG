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

## 独立公开基准：MultiHop-RAG

本地中文语料评测和公开 MultiHop-RAG 基准严格分离：前者继续使用 `eval` 与 `config/default.yaml`；后者使用自己的公开英文语料、索引、配置和运行产物，绝不写入 `data/raw`、`data/index` 或本地题集。

```powershell
# 首次下载公开数据、转换为 Markdown，并生成可复核的 Gold 证据清单。
python -m base_rag prepare-multihop

# 用独立索引建立公开基准语料库。
python -m base_rag ingest --config config/multihoprag.yaml

# 默认比较全部五种 Profile；只执行检索，不调用最终答案生成。
python -m base_rag eval-multihop

# 先以 20 题验证某一个 Profile；不会影响完整结果目录。
python -m base_rag eval-multihop --profile advanced --limit 20
```

`prepare-multihop` 下载 [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) 的 `corpus.json` 与 `MultiHopRAG.json`，记录两个文件的 SHA-256，并将每篇文档的标题、作者、来源、发布时间、类别、URL 和正文写入独立 Markdown 语料。`--force` 才会重新下载和转换已准备的基准。

`eval-multihop` 的结果写入 `runs/multihoprag/evaluations/<时间戳>/`。报告同时给出两类指标：严格的 `Evidence Coverage@4/@10` 与 `Complete Evidence@4/@10` 要求覆盖每题全部 Gold 事实；`Hits@4/@10`、`MAP@10`、`MRR@10` 使用公开仓库的字符串包含式检索口径，便于横向对照。该公开基准是英文跨文档检索评测，不等同于本地中文知识库效果，也不包含端到端答案正确率。

Query Rewrite 通过 `query_rewrite.language` 选择指令语言：本地配置固定为 `zh`，MultiHop-RAG 固定为 `en`；如语料本身混合多种语言，可设为 `auto`，它会按问题中是否含中文字符选择对应指令。三种取值均要求改写结果保持问题原语言。

`prepare-multihop` 和 `ingest` 会显示阶段进度条；`eval` 与 `eval-multihop` 会按 Profile 显示题目完成进度、成功数与失败数。

评测默认在同一个 Profile 内并发执行 4 道题；并发数可通过 `evaluation.concurrency` 调整，设为 `1` 即恢复串行。并发模式显示线程安全的题目总进度，串行模式额外显示当前题目的阶段进度。FAISS/BM25 索引在每个 Profile 内只加载一次，结果始终按题集原顺序保存。

Embedding、生成和 Rerank 请求遇到连接失败、超时、HTTP 408/409/425/429 或 5xx 时，会按照 `models.max_retries` 自动重试，并在每次重试前固定等待 `models.retry_delay_seconds`（默认 5 秒）。鉴权失败、参数错误等不可恢复请求立即失败。单题最终失败后仍会记录题号、失败阶段和原始错误并继续评测；失败题按 0 分计入检索指标，并在汇总中单独可查。

## Phase 3：受控 Agentic Multi-Hop 检索

Phase 3 基于 LangGraph 状态图构建了受控多跳检索闭环：包含 Query Router（单/多跳路由）、逐跳 Hybrid-Rerank 检索、Evidence Grader（证据充分性判断）、按需 Query Corrector（单跳检索失败纠错）与轮询证据交织截取。

```powershell
# 单题 Agentic 检索（返回完整多跳命中与链路追踪 Trace）
python -m base_rag agentic-retrieve --config config/default.yaml --question "创办 A 公司的人后来领导了哪家机构？"

# MultiHop-RAG 分层开发集 (60题) 评测与 Baseline 对照
python -m base_rag agentic-eval --config config/multihoprag.yaml --split dev --system both

# MultiHop-RAG 锁定测试集 (120题) 评测
python -m base_rag agentic-eval --config config/multihoprag.yaml --split test --system both
```

评测产物写入 `runs/multihoprag/agentic_evaluations/<split>-<时间戳>/`，自动生成包含成对增量（Delta）、95% 置信区间、平均跳数与终止原因分布的 `REPORT.md` 与 `summary.json`。支持题目级 checkpoint 自动断点恢复。
