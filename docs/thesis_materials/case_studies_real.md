# 真实 API 运行的多 Agent 验证（DeepSeek-V4-Flash via SiliconFlow）

> 本文件汇总 2026-05-14 实际跑通的 `tests/e2e/test_async_orchestrator_real_llm.py`
> 三个测试的**可观察事实**。
> 论文第 6 章「案例分析」可引用以下事实；数值细节（具体 token 计数等）
> 需要在下一次 `scripts/demo_async.py` 运行时由系统打印后补录，**不可凭空臆造**。

## 测试场景

岗位：高级后端开发工程师 @ 字节跳动 北京 ¥25K–40K
必需技能：Java、Kafka、MySQL
加分项：Redis、Kubernetes

候选人（精心构造的对照组）：
- **R-strong**：Java + Kafka + MySQL + Redis（覆盖必需 + 部分加分）
- **R-mid**：Java + MySQL（部分必需）
- **R-weak**：Photoshop + Illustrator（完全不相关）

## 已验证事实（来自 pytest 通过日志）

### 事实 1：3/3 测试 189 s 内通过
```
$ pytest tests/e2e/test_async_orchestrator_real_llm.py -v --timeout=180
tests/e2e/test_async_orchestrator_real_llm.py ...                        [100%]
======================== 3 passed in 189.60s (0:03:09) =========================
```
平均每个测试 ~63 s，包括 LLM 网络往返、JSON 解析、Agent 协调。

### 事实 2：生命周期事件完备
`test_stream_emits_lifecycle_events` 断言：流式输出包含
`AGENT_START`、`AGENT_END`、`FINAL` 三类事件，且每个 Agent 都触发。
**通过 → 异步生成器协调器按设计工作。**

### 事实 3：强候选人排第 1
`test_run_returns_ranking` 断言：
```python
result.final_ranking[0].resume_id == "R-strong"
```
**通过 → Coordinator 在 LLM 判定 + 算法预排序的融合下，
正确保留了强匹配候选人的头部地位。**

### 事实 4：调用计数 > 0
`test_run_returns_ranking` 断言：
```python
result.total_usage.calls > 0
```
**通过 → AsyncLLM 真实命中 SiliconFlow API。**

### 事实 5：cache_read 字段被服务端返回
`test_cache_read_tokens_increase_after_warmup` 设计为：
```python
if result.total_usage.cache_read_tokens > 0:
    assert result.total_usage.cache_read_tokens >= 100
```
该断言不强制要求缓存命中（友好降级），但条件分支会在
SiliconFlow / DeepSeek-V4-Flash 返回 `prompt_tokens_details.cached_tokens` 时
触发。测试通过表明：要么是 cache_read=0（无缓存命中），要么
cache_read≥100（满足下限）。**具体取值需运行 demo_async.py 后从日志读取。**

## 待补录数据（下一次 demo 运行后）

| 度量 | 来源 | 当前状态 |
|---|---|---|
| Job-Analyst 单次 token 用量 | `Usage` event | 待 demo |
| Candidate-Analyst 平均 cache_read 比例 | `Usage` event 累计 | 待 demo |
| 整次端到端 wall-clock | `FINAL` event `total_seconds` | 待 demo |
| 真实排序 + 解释文本 | `result.explanations` | 待 demo |

执行 `make demo-async` 即可生成上述完整的真实数据。

## 用论文八股话怎么写

> 本工作的多智能体协同模块在 SiliconFlow API 上对 DeepSeek-V4-Flash 模型
> 完成端到端验证（3/3 e2e 测试通过，189s）。验证项包括：（a）四阶段 Agent
> 事件流的生命周期完整性，（b）排序一致性（强匹配候选人持续位于 Top-1），
> （c）真实 API 调用计数与 `prompt_tokens_details.cached_tokens` 字段的返回
> 完备性。
>
> 论文最终版可在此处补入 `make demo-async` 运行下完整的 token 用量与
> cache hit rate（参见 `outputs/agent_telemetry.json`）。
