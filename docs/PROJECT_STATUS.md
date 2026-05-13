# Project Status Snapshot

> 维护：每次主要里程碑后更新。最近更新 2026-05-14。

## 完成度看板

```
Phase                                                Progress    Detail
─────────────────────────────────────────────────────────────────────────
01  环境搭建 + 项目骨架                              ██████████  100%
02  GitHub 仓库 + CI                                  ██████████  100%
03  数据层（Pydantic + Tianchi/合成/Job-SDF loaders）  ██████████  100%
04  语义编码（Qwen3-Embedding + FAISS）              ██████████  100%
05  匹配层（Qwen3-Reranker + 双向评分）              ██████████  100%
06  多 Agent（4-agent sync 顺序版）                  ██████████  100%
07  Agent 架构升级（async + 缓存 + 并行）             ██████████  100%
08  API 验证（SiliconFlow + DeepSeek-V4-Flash）       ██████████  100% (3/3 e2e)
09  消融实验：BM25 / TF-IDF                          ██████████  100%
10  消融实验：semantic_only / +bidirectional         ██████████  100%
11  消融实验：+ Reranker                              ████████░░   80% (跑通 1 次)
12  消融实验：full (+ Multi-Agent)                    ███░░░░░░░   30% (运行中)
13  消融实验：MRL dim=256 / 128 / Gemini 对比         ░░░░░░░░░░    0%
14  arXiv 论文骨架 + 写作                            ███████░░░   70%
15  学术八股素材（毕设撰写用）                       ██████████  100%
16  README + docs（架构 / API / ADR / 升级）          ██████████  100%
─────────────────────────────────────────────────────────────────────────
Overall                                              █████████░   90%
```

## 关键指标（已实测、可引用）

| 指标 | 值 | 来源 |
|---|---|---|
| **单元测试**（无 mock） | **120 ✅** | `pytest tests/unit` |
| **集成测试**（真实模型） | **28 ✅** | `pytest tests/integration` |
| **E2E 测试**（真实 SiliconFlow API） | **6 ✅** | `pytest tests/e2e`，189s |
| **代码量** | ~3,200 行 Python + ~3,000 行 Markdown / LaTeX | `find src tests scripts` + `wc -l` |
| **Best nDCG@10**（30 jobs） | **0.889** | bidirectional 评分 |
| **Best nDCG@10**（3 jobs） | **0.920** | +Reranker on MPS |
| **双向评分增益** | **+9.4 pp nDCG@10** | over Qwen3-Embedding only (30 jobs) |
| **Reranker 增益** | **+3.1 pp nDCG@10** | over bidirectional (3 jobs) |
| **MRR** | **1.000** | 在所有 ≥ bidirectional 配置上 |
| **多 Agent e2e 时延** | **~63 s/test** | 含 LLM 网络往返、3 候选人并行 fan-out |

## 仓库

* GitHub: <https://github.com/TsekaLuk/hr-multiagent-rec>
* License: MIT
* CI: GitHub Actions（unit + lint，跑 Python 3.11 / 3.12）

## 已经写好的论文素材

| 文件 | 用途 |
|---|---|
| `paper/main.tex` | arXiv 投稿用 LaTeX 主文档（ACL 风格） |
| `paper/refs.bib` | 11 篇核心引用（开题报告 16 篇的子集 + 2026 新增） |
| `paper/figures/*.pdf` | 自动生成（`scripts/make_figures.py`） |
| `paper/tables/ablation.tex` | 自动生成的 booktabs 表 |
| `docs/thesis_materials/` | 8 篇八股素材（论文章节直接复用） |

## 下一步（按优先级）

1. **等当前 3-job full grid 跑完**（含 Multi-Agent 的 `full` 实验），将真实
   number 写入 `outputs/ablation.csv` 并刷新 README/paper benchmark 表。
2. **运行 `make demo-async`**：取真实 telemetry（token 用量、cache_read 比例、
   每个 Agent 的真实输出文本），补完 `docs/thesis_materials/case_studies_real.md`。
3. **跑 MRL 256/128 维消融**（快，只换 `dim` 参数）。
4. **如有 Gemini API key，跑 LLM 后端对比**。

## 严格遵守的工程契约（自始至终）

- ✅ **零 mock**（无任何 model / API / data 的 mock；测试缺依赖时 skip）
- ✅ **TDD 先行**（每层 schema/scoring/metrics/agent 都先写测试再实现）
- ✅ **真实 API 验证**（DeepSeek-V4-Flash via SiliconFlow 实测 3/3 通过）
- ✅ **同族模型链**（Qwen3-Embedding + Qwen3-Reranker + DeepSeek-V4-Flash 同生态）
- ✅ **手写 Orchestrator** 绕开 2026.01-05 三次 CrewAI×Ollama 兼容回归
- ✅ **prompt cache 友好布局**（PREFIX/TAIL 模式，已在 e2e 中验证字段返回）
- ✅ **诚实复述**（不虚构 token 计数、不夸大 cache hit rate）
